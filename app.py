import os
import re
import time
import requests
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone

# -----------------------------
# 기본 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="YouTube 키워드 트렌드 분석기",
    page_icon="📈",
    layout="wide"
)

# -----------------------------
# 상수 / 전역 패턴
# -----------------------------
BASE_URL = "https://www.googleapis.com/youtube/v3"
KST = timezone(timedelta(hours=9))

# 분석에 큰 의미가 없는 단어들(불용어)
DEFAULT_STOPWORDS = {
    "영상", "동영상", "브이로그", "vlog",
    "the", "a", "to", "of", "in", "on", "for", "and", "is", "are", "with", "from",
    "한", "것", "수", "이", "그", "저", "및", "등", "때", "때문",
    "오늘", "하루", "일상",
    "채널", "유튜브", "youtube",
    "shorts", "short",
    "공식", "official", "full",
    "2023", "2024", "2025"
}

# 한글/영문 단어 토큰 추출용 정규식
TOKEN_PATTERN = re.compile(r"[A-Za-z가-힣]+")

# -----------------------------
# API 키 로딩 유틸
# -----------------------------
def load_api_key():
    """
    Streamlit secrets > OS 환경변수(YOUTUBE_API_KEY) 순서로 로드.
    둘 다 없으면 빈 문자열 반환.
    """
    key_from_secrets = st.secrets.get("YOUTUBE_API_KEY", None)
    if key_from_secrets:
        return key_from_secrets
    return os.getenv("YOUTUBE_API_KEY", "")

API_KEY = load_api_key()


# -----------------------------
# YouTube API 호출 유틸
# -----------------------------
def yt_get(path, params, sleep=0.0):
    """
    YouTube Data API GET 래퍼.
    - API 키 주입
    - 간단 쿼터 보호용 sleep
    - 오류 시 RuntimeError 발생
    """
    if not API_KEY:
        raise RuntimeError(
            "API 키가 없습니다. Streamlit Cloud의 'Manage app → Secrets'에 "
            'YOUTUBE_API_KEY = "실제_API_KEY" 형식으로 등록하거나, '
            "로컬에서는 OS 환경변수 YOUTUBE_API_KEY를 설정하세요."
        )

    final_params = dict(params)
    final_params["key"] = API_KEY
    url = f"{BASE_URL}/{path}"

    r = requests.get(url, params=final_params, timeout=30)
    if sleep:
        time.sleep(sleep)

    if r.status_code != 200:
        raise RuntimeError(f"API 오류 {r.status_code}: {r.text}")
    return r.json()


def youtube_search(keyword, published_after, published_before,
                   region_code, max_results=50):
    """
    search.list를 사용해서 videoId 목록을 모은다.

    keyword: 검색어
    published_after / published_before: ISO8601(UTC) 문자열
    region_code: "KR", "US" 등 지역 코드 (빈 문자열이면 전체)
    max_results: 최종적으로 가져올 목표 개수
    """
    ids = []
    fetched = 0
    next_page_token = None

    while fetched < max_results:
        page_size = min(50, max_results - fetched)

        query_params = {
            "part": "id",
            "type": "video",
            "q": keyword,
            "publishedAfter": published_after,
            "publishedBefore": published_before,
            "maxResults": page_size,
            "order": "relevance",
        }
        # regionCode는 선택 항목. 빈 문자열이면 넣지 않는다.
        if region_code:
            query_params["regionCode"] = region_code

        # 다음 페이지가 있으면 pageToken 추가
        if next_page_token:
            query_params["pageToken"] = next_page_token

        data = yt_get("search", query_params, sleep=0.05)

        items = data.get("items", [])
        for item in items:
            vid = item.get("id", {}).get("videoId")
            if vid:
                ids.append(vid)

        fetched += len(items)
        next_page_token = data.get("nextPageToken")

        # 다음 페이지가 없거나 더 이상 결과가 없으면 중단
        if (not next_page_token) or (len(items) == 0):
            break

    # dict.fromkeys: 순서를 유지하면서 중복 제거
    return list(dict.fromkeys(ids))


def youtube_videos_stats(video_ids):
    """
    videos.list를 사용해 메타데이터/통계를 DataFrame으로 변환한다.
    반환 컬럼 예:
    - videoId, title, description, channelTitle, publishedAt
    - viewCount, likeCount, commentCount
    - ER(%) = (like+comment)/view * 100
    """
    rows = []

    for i in range(0, len(video_ids), 50):
        chunk = ",".join(video_ids[i:i+50])
        data = yt_get(
            "videos",
            {
                "part": "snippet,statistics,contentDetails",
                "id": chunk,
            },
            sleep=0.05,
        )

        for it in data.get("items", []):
            s = it.get("snippet", {})
            stats = it.get("statistics", {})
            row = {
                "videoId": it.get("id"),
                "title": s.get("title", ""),
                "description": s.get("description", ""),
                "channelTitle": s.get("channelTitle", ""),
                "publishedAt": s.get("publishedAt", ""),
                "viewCount": int(stats.get("viewCount", 0) or 0),
                "likeCount": int(stats.get("likeCount", 0) or 0),
                "commentCount": int(stats.get("commentCount", 0) or 0),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty:
        # publishedAt을 시계열로 변환
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")

        # 참여율(ER, Engagement Rate) 계산
        df["ER(%)"] = np.where(
            df["viewCount"] > 0,
            (df["likeCount"] + df["commentCount"]) / df["viewCount"] * 100.0,
            0.0,
        ).round(3)

    return df


# -----------------------------
# 텍스트 처리 / 키워드 추출
# -----------------------------
def tokenize(text: str):
    """
    제목+설명에서 한글/영문 단어만 추출하고
    불용어와 너무 짧은 단어는 제거한다.
    """
    tokens = TOKEN_PATTERN.findall(text.lower())
    return [
        t for t in tokens
        if t not in DEFAULT_STOPWORDS and len(t) >= 2
    ]


def keywords_from_df(df, topn=100):
    """
    DataFrame에서 title+description을 모아 토큰화하고
    상위 빈도 키워드를 (단어, 빈도) 리스트로 반환한다.
    """
    corpus = []
    for _, row in df.iterrows():
        title_txt = row.get("title", "")
        desc_txt = row.get("description", "")
        corpus.extend(tokenize(f"{title_txt} {desc_txt}"))

    counter = Counter(corpus)
    return counter.most_common(topn)


def draw_wordcloud(freqs, font_path=None):
    """
    (단어, 빈도) or Counter 기반으로 워드클라우드를 그린다.
    font_path: 한글 폰트 경로 (Cloud에는 기본 한글 폰트가 없을 수 있음)
    """
    wc = WordCloud(
        width=900,
        height=500,
        background_color="white",
        font_path=font_path if font_path else None,
        collocations=False
    )
    wc.generate_from_frequencies(dict(freqs))

    fig = plt.figure(figsize=(9, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    return fig


# -----------------------------
# UI 헤더
# -----------------------------
st.title("📈 유튜브 키워드 트렌드 분석기")
st.caption("키워드/기간으로 상위 동영상을 수집해 워드클라우드와 참여율(ER)을 분석합니다.")

# -----------------------------
# 사이드바 입력 UI
# -----------------------------
with st.sidebar:
    st.subheader("검색 설정")

    keyword = st.text_input(
        "키워드 (예: 브이로그 / 영어 가능)",
        value="브이로그"
    )

    days = st.number_input(
        "최근 N일",
        min_value=1,
        max_value=365,
        value=30,
        step=1
    )

    max_results = st.slider(
        "수집할 영상 수",
        min_value=10,
        max_value=200,
        value=80,
        step=10
    )

    region_code = st.text_input(
        "지역 코드 (선택, KR/US/JP 등)",
        value="KR"
    ).strip().upper()

    font_path = st.text_input(
        "워드클라우드 한글 폰트 경로(선택)",
        value=""
    )

    st.markdown("---")

    # API 키 상태 표시
    if API_KEY:
        st.success("YouTube API 키 로드 완료")
    else:
        st.error("YouTube API 키가 없습니다.")
        st.markdown(
            "Streamlit Cloud → **Manage app → Secrets** 에 아래처럼 등록하세요:\n\n"
            "```toml\n"
            'YOUTUBE_API_KEY = "여기에_실제_API_KEY"\n'
            "```\n\n"
            "로컬(윈도우 PowerShell)에서 테스트할 경우:\n\n"
            "```powershell\n"
            'setx YOUTUBE_API_KEY "여기에_실제_API_KEY"\n'
            "```\n\n"
            "로컬(macOS / Linux)에서는:\n\n"
            "```bash\n"
            'export YOUTUBE_API_KEY="여기에_실제_API_KEY"\n'
            "```"
        )

    run = st.button(
        "데이터 수집/분석 실행",
        type="primary",
        use_container_width=True
    )

# -----------------------------
# 분석용 기간 계산 (UTC 기준)
# -----------------------------
now_utc = datetime.now(timezone.utc)
published_after = (now_utc - timedelta(days=int(days))).isoformat()
published_before = now_utc.isoformat()

# -----------------------------
# 실행 로직
# -----------------------------
if run:
    # API 키 없으면 여기서 막고 안내
    if not API_KEY:
        st.warning("API 키가 없어 YouTube API 호출을 중단합니다. 키를 설정한 뒤 다시 실행해주세요.")
        st.stop()

    try:
        with st.spinner("검색 중…"):
            ids = youtube_search(
                keyword=keyword,
                published_after=published_after,
                published_before=published_before,
                region_code=region_code,
                max_results=max_results
            )

        if len(ids) == 0:
            st.warning("검색 결과가 없습니다. 키워드/기간/지역 코드를 조정해보세요.")
        else:
            with st.spinner("영상 메타데이터/통계 수집 중…"):
                df = youtube_videos_stats(ids)

            if df.empty:
                st.warning("수집된 통계가 없습니다.")
            else:
                st.success(f"수집 완료: {len(df)}개 영상")

                # KPI 카드
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("총 조회수", f"{df['viewCount'].sum():,}")
                with c2:
                    st.metric("평균 ER(%)", f"{df['ER(%)'].mean():.2f}")
                with c3:
                    st.metric("평균 댓글 수", f"{df['commentCount'].mean():.1f}")
                with c4:
                    st.metric("분석 기간(일)", f"{days}")

                # 상위 영상 테이블 (ER 내림차순 우선, 그다음 조회수)
                st.subheader("상위 영상 (ER 내림차순)")
                df_sorted = df.sort_values(
                    ["ER(%)", "viewCount"],
                    ascending=[False, False]
                ).reset_index(drop=True)

                st.dataframe(
                    df_sorted[[
                        "title",
                        "channelTitle",
                        "viewCount",
                        "likeCount",
                        "commentCount",
                        "ER(%)",
                        "publishedAt",
                        "videoId"
                    ]],
                    use_container_width=True,
                    height=360
                )

                # 워드클라우드
                st.subheader("워드클라우드 (제목+설명 기반)")
                freqs = keywords_from_df(df_sorted, topn=120)
                if len(freqs) == 0:
                    st.info("유의미한 키워드가 부족합니다. 불용어를 줄이거나 기간/영상 수를 늘려보세요.")
                else:
                    fig = draw_wordcloud(
                        freqs,
                        font_path=font_path.strip() if font_path.strip() else None
                    )
                    st.pyplot(fig, clear_figure=True)

                # 키워드 빈도 TOP 30
                st.subheader("키워드 상위 빈도")
                top_k = pd.DataFrame(freqs[:30], columns=["keyword", "freq"])
                st.dataframe(
                    top_k,
                    use_container_width=True,
                    height=400
                )

                # 참여율 vs 조회수 산포
                st.subheader("참여율(ER%) vs 조회수")
                st.scatter_chart(
                    df_sorted,
                    x="viewCount",
                    y="ER(%)",
                    size="commentCount",
                    color=None
                )
                st.caption(
                    "버블 크기는 댓글 수. "
                    "ER(%) = (좋아요 수 + 댓글 수) / 조회수 × 100"
                )

                # CSV 다운로드
                st.download_button(
                    label="CSV 다운로드",
                    data=df_sorted.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"yt_{keyword}_{days}d.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    except Exception as e:
        st.error(f"오류: {e}")
        st.info("API 키 설정, 기간(days), 지역 코드(KR/US/JP 등), 또는 YouTube API 쿼터 상태를 다시 확인하세요.")

# -----------------------------
# 도움말 / 주의사항
# -----------------------------
with st.expander("도움말 / 주의사항"):
    st.markdown(r'''
- **ER(%)** = (likeCount + commentCount) / viewCount × 100  
  - 일부 채널은 좋아요 수를 숨기거나 댓글을 막아 둘 수 있어서 0으로 나타날 수 있습니다.

- **지역 코드(regionCode)**  
  - `"KR"` 같이 두 글자의 국가 코드를 쓰면 지역별 검색 경향을 더 반영할 수 있습니다.
  - 빈칸으로 두면 전세계 기준으로 가져옵니다.

- **워드클라우드 한글 폰트 경로**  
  - 예: `C:/Windows/Fonts/malgun.ttf` (로컬 Windows 환경의 맑은 고딕)
  - Streamlit Cloud(Ubuntu 기반)에서는 기본 한글 폰트가 없을 수 있어, 직접 .ttf 경로를 제공해야 할 수 있습니다.

- **쿼터 제한**  
  - YouTube Data API v3에는 일일 쿼터가 있습니다.
  - 너무 많은 영상을 짧은 기간에서 가져오면 403/429 계열 에러가 날 수 있습니다.
    → 기간(days)을 늘리거나, max_results를 줄이세요.
''')
