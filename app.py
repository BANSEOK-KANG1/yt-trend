import os
import re
import time
import math
import json
import requests
import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone

# -----------------------------
# 환경 설정
# -----------------------------
st.set_page_config(page_title="YouTube 키워드 트렌드 분석기", page_icon="📈", layout="wide")
API_KEY = st.secrets.get("YOUTUBE_API_KEY", os.getenv("YOUTUBE_API_KEY", ""))

BASE_URL = "https://www.googleapis.com/youtube/v3"
KST = timezone(timedelta(hours=9))
DEFAULT_STOPWORDS = {
    # 한국어/영어 공통 불용어(필요 시 추가)
    "영상","동영상","브이로그","vlog","the","a","to","of","in","on","for","and","is","are","with","from",
    "한","것","수","이","그","저","및","등","때","때문","오늘","하루","일상","채널","유튜브","youtube",
    "shorts","short","공식","official","full","2023","2024","2025"
}

# 한글/영문 단어만 남기기 위한 간단 토크나이저
TOKEN_PATTERN = re.compile(r"[A-Za-z가-힣]+")

# -----------------------------
# 유틸
# -----------------------------
def yt_get(path, params, sleep=0.0):
    """YouTube Data API GET 래퍼(간단 쿼터/속도 제어)."""
    if not API_KEY:
        raise RuntimeError("API 키가 없습니다. .streamlit/secrets.toml 또는 환경변수 YOUTUBE_API_KEY를 설정하세요.")
    params = dict(params)
    params["key"] = API_KEY
    url = f"{BASE_URL}/{path}"
    r = requests.get(url, params=params, timeout=30)
    if sleep:
        time.sleep(sleep)
    if r.status_code != 200:
        raise RuntimeError(f"API 오류 {r.status_code}: {r.text}")
    return r.json()

def youtube_search(keyword, published_after, published_before, region_code, max_results=50):
    """
    검색 결과에서 videoId들 수집. (search.list)
    """
    ids = []
    page_token = None
    fetched = 0
    while fetched < max_results:
        page_size = min(50, max_results - fetched)
        data = yt_get(
            "search",
            {
                "part": "id",
                "type": "video",
                "q": keyword,
                "publishedAfter": published_after,
                "publishedBefore": published_before,
                "regionCode": region_code or "",
                "maxResults": page_size,
                "order": "relevance",
            },
            sleep=0.05,
        )
        for item in data.get("items", []):
            vid = item.get("id", {}).get("videoId")
            if vid:
                ids.append(vid)
        fetched += len(data.get("items", []))
        page_token = data.get("nextPageToken")
        if not page_token or len(data.get("items", [])) == 0:
            break
        # 다음 페이지를 명시적으로 쓰고 싶다면 params에 pageToken 추가 (여긴 relevance 우선이라 생략)

    return list(dict.fromkeys(ids))  # 중복 제거

def youtube_videos_stats(video_ids):
    """
    videos.list로 메타데이터/통계를 조회.
    """
    rows = []
    # 50개씩 끊어서 조회
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
                "title": s.get("title",""),
                "description": s.get("description",""),
                "channelTitle": s.get("channelTitle",""),
                "publishedAt": s.get("publishedAt",""),
                "viewCount": int(stats.get("viewCount", 0) or 0),
                "likeCount": int(stats.get("likeCount", 0) or 0),  # 일부 채널은 0 또는 비공개
                "commentCount": int(stats.get("commentCount", 0) or 0),
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["publishedAt"] = pd.to_datetime(df["publishedAt"])
        df["ER(%)"] = np.where(
            df["viewCount"] > 0,
            (df["likeCount"] + df["commentCount"]) / df["viewCount"] * 100.0,
            0.0,
        ).round(3)
    return df

def tokenize(text):
    tokens = TOKEN_PATTERN.findall(text.lower())
    return [t for t in tokens if t not in DEFAULT_STOPWORDS and len(t) >= 2]

def keywords_from_df(df, topn=100):
    corpus = []
    for _, row in df.iterrows():
        corpus.extend(tokenize(row.get("title","") + " " + row.get("description","")))
    counter = Counter(corpus)
    return counter.most_common(topn)

def draw_wordcloud(freqs, font_path=None):
    wc = WordCloud(
        width=900, height=500,
        background_color="white",
        font_path=font_path, # 한글 폰트 경로 지정 필요 시
        collocations=False
    )
    wc.generate_from_frequencies(dict(freqs))
    fig = plt.figure(figsize=(9,5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    return fig

# -----------------------------
# UI
# -----------------------------
st.title("📈 유튜브 키워드 트렌드 분석기")
st.caption("키워드/기간으로 상위 동영상을 수집해 워드클라우드와 참여율(ER)을 분석합니다.")

with st.sidebar:
    st.subheader("검색 설정")
    keyword = st.text_input("키워드 (예: 브이로그 / 영어 가능)", "브이로그")
    days = st.number_input("최근 N일", min_value=1, max_value=365, value=30, step=1)
    max_results = st.slider("수집할 영상 수", min_value=10, max_value=200, value=80, step=10)
    region_code = st.text_input("지역 코드 (선택, KR/US/JP 등)", "KR").strip().upper()
    font_path = st.text_input("워드클라우드 한글 폰트 경로(선택)", "")
    st.markdown("---")
    run = st.button("데이터 수집/분석 실행", type="primary", use_container_width=True)

# 기간 계산 (UTC 기준 ISO8601)
now = datetime.now(timezone.utc)
published_after = (now - timedelta(days=int(days))).isoformat()
published_before = now.isoformat()

# 결과 캐시 컨테이너
result_area = st.container()

if run:
    try:
        with st.spinner("검색 중…"):
            ids = youtube_search(keyword, published_after, published_before, region_code, max_results=max_results)
        if len(ids) == 0:
            st.warning("검색 결과가 없습니다. 키워드/기간/지역 코드를 조정해보세요.")
        else:
            with st.spinner("영상 메타데이터/통계 수집 중…"):
                df = youtube_videos_stats(ids)

            if df.empty:
                st.warning("수집된 통계가 없습니다.")
            else:
                st.success(f"수집 완료: {len(df)}개 영상")

                # 상단 KPI
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("총 조회수", f"{df['viewCount'].sum():,}")
                with c2:
                    st.metric("평균 ER(%)", f"{df['ER(%)'].mean():.2f}")
                with c3:
                    st.metric("평균 댓글 수", f"{df['commentCount'].mean():.1f}")
                with c4:
                    st.metric("분석 기간(일)", f"{days}")

                # 테이블
                st.subheader("상위 영상(ER 내림차순)")
                df_sorted = df.sort_values(["ER(%)","viewCount"], ascending=[False, False]).reset_index(drop=True)
                st.dataframe(
                    df_sorted[["title","channelTitle","viewCount","likeCount","commentCount","ER(%)","publishedAt","videoId"]],
                    use_container_width=True, height=360
                )

                # 워드클라우드
                st.subheader("워드클라우드 (제목+설명)")
                freqs = keywords_from_df(df_sorted, topn=120)
                if len(freqs) == 0:
                    st.info("유의미한 키워드가 부족합니다. 불용어를 줄이거나 기간/영상 수를 늘려보세요.")
                else:
                    fig = draw_wordcloud(freqs, font_path=font_path if font_path.strip() else None)
                    st.pyplot(fig, clear_figure=True)

                # 빈도 상위 표
                st.subheader("키워드 상위 빈도")
                top_k = pd.DataFrame(freqs[:30], columns=["keyword","freq"])
                st.dataframe(top_k, use_container_width=True, height=400)

                # 참여율 vs 조회수 산포
                st.subheader("참여율(ER%) vs 조회수")
                st.scatter_chart(df_sorted, x="viewCount", y="ER(%)", size="commentCount", color=None)
                st.caption("버블 크기는 댓글 수. ER = (좋아요 + 댓글) / 조회수 × 100")

                # CSV 내보내기
                st.download_button(
                    "CSV 다운로드",
                    df_sorted.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"yt_{keyword}_{days}d.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    except Exception as e:
        st.error(f"오류: {e}")
        st.info("API 키, 기간, 지역 코드, 쿼터 상태를 확인하세요.")

# 도움말
with st.expander("도움말 / 주의사항"):
    st.markdown("""
- **ER(%)** = (likeCount + commentCount) / viewCount × 100  
  - 일부 채널은 **좋아요 수 비공개**로 0으로 올 수 있어요.
- **지역 코드**는 선택입니다. 글로벌 트렌드를 보려면 빈칸으로 두세요.
- **워드클라우드 한글 폰트**가 필요하면 시스템 폰트(.ttf) 경로를 입력하세요.  
  - 예: `C:/Windows/Fonts/malgun.ttf` (맑은고딕)
- 쿼터 초과 시 오류가 날 수 있어요. 기간/수집량을 낮추세요.
""")
