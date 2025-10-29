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

# 불용어(분석에 의미가 적은 단어들 제거)
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
    Streamlit secrets > 환경변수(YOUTUBE_API_KEY) 순서로 로드.
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
    - 오류시 RuntimeError로 올림
    """
    if not API_KEY:
        raise RuntimeError(
            "API 키가 없습니다. Streamlit Cloud의 'Manage app → Secrets'에 "
            'YOUTUBE_API_KEY = "실제키" 형식으로 등록하거나, '
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
    검색(search.list)으로 videoId 목록 수집.
    keyword: 검색어
    published_after/before: ISO8601 UTC 시간 문자열
    region_code: 'KR', 'US' 등의 지역 코드 (빈 문자열 가능)
    max_results: 최종적으로 가져올 최대 비디오 수
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
        # regionCode는 선택값이지만, 빈 문자열이면 에러 줄 수 있으므로 조건부 추가
        if region_code:
            query_params["regionCode"] = region_code

        # pageToken이 있으면 추가
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

        # 다음 페이지가 없거나 이번 페이지에서 아무 것도 못 가져왔으면 종료
        if (not next_page_token) or (len(items) == 0):
            break

    # dict.fromkeys: 입력 순서를 유지하면서 중복 제거
    return list(dict.fromkeys(ids))


def youtube_videos_stats(video_ids):
    """
    videos.list 로 비디오 메타데이터/통계를 DataFrame으로 변환.
    - title, description, channelTitle, publishedAt
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
        # 문자열을 시계열로 변환
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")

        # ER (Engagement Rate)
        df["ER(%)"] = np.where(
            df["viewCount"] > 0,
            (df["likeCount"] + df["commentCount"]) / df["viewCount"] * 100.0,
            0.0,
        ).round(3)

    return df


# -----------------------------
# 텍스트 처리 / 키워드 분석
# -----------------------------
def tokenize(text: str):
    """
    제목+설명에서 한글/영문 단어만 추출하고,
    불용어/길이<2 단어 제거.
    """
    tokens = TOKEN_PATTERN.findall(text.lower())
    return [
        t for t in tokens
        if t not in DEFAULT_STOPWORDS and len(t) >= 2
    ]


def keywords_from_df(df, topn=100):
    """
    DataFrame에서 title+description을 합쳐 토큰화하고
    상위 빈도 키워드 (topn개) 반환.
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
    단어 빈도(freqs)를 기반으로 워드클라우드 그려서 matplotlib Figure 반환.
    font_path: 한글 폰트 경로 필요 시 지정.
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
# UI - 헤더
# -----------------------------
st.title("📈 유튜브 키워드 트렌드 분석기")
st.caption("키워드/기간으로 상위 동영상을 수집해 워드클라우드와 참여율(ER)을 분석합니다.")

# -----------------------------
# UI - 사이드바 입력
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
        st.markdown("""
**Streamlit Cloud > Manage app > Secrets** 에 아래처럼 등록하세요:

```toml
YOUTUBE_API_KEY = "여기에_실제_API_KEY"
