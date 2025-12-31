# =============================
# Imports (Streamlit 명령 전)
# =============================
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
import matplotlib.font_manager as fm
import seaborn as sns
from datetime import datetime, timedelta, timezone

# =============================
# 반드시 가장 먼저 실행
# =============================
st.set_page_config(
    page_title="YouTube 키워드 트렌드 분석기",
    page_icon="📈",
    layout="wide"
)

# =============================
# 한글 폰트 로딩 (Cloud 대응)
# =============================
@st.cache_resource
def load_font():
    font_path = os.path.join("fonts", "Pretendard-Regular.otf")
    font_prop = fm.FontProperties(fname=font_path)

    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    sns.set(font=font_prop.get_name())

    return font_prop

font_prop = load_font()

# =============================
# 상수 / 설정
# =============================
BASE_URL = "https://www.googleapis.com/youtube/v3"
KST = timezone(timedelta(hours=9))

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

TOKEN_PATTERN = re.compile(r"[A-Za-z가-힣]+")

# =============================
# API Key 로딩
# =============================
def load_api_key():
    return st.secrets.get("YOUTUBE_API_KEY") or os.getenv("YOUTUBE_API_KEY", "")

API_KEY = load_api_key()

# =============================
# YouTube API Wrapper
# =============================
def yt_get(path, params, sleep=0.0):
    if not API_KEY:
        raise RuntimeError("YouTube API Key가 설정되지 않았습니다.")

    params = dict(params)
    params["key"] = API_KEY

    r = requests.get(f"{BASE_URL}/{path}", params=params, timeout=30)
    if sleep:
        time.sleep(sleep)

    if r.status_code != 200:
        raise RuntimeError(f"API 오류 {r.status_code}: {r.text}")

    return r.json()

def youtube_search(keyword, published_after, published_before, region_code, max_results):
    ids, fetched, token = [], 0, None

    while fetched < max_results:
        size = min(50, max_results - fetched)
        params = {
            "part": "id",
            "type": "video",
            "q": keyword,
            "publishedAfter": published_after,
            "publishedBefore": published_before,
            "maxResults": size,
            "order": "relevance"
        }
        if region_code:
            params["regionCode"] = region_code
        if token:
            params["pageToken"] = token

        data = yt_get("search", params, 0.05)
        items = data.get("items", [])

        for it in items:
            vid = it.get("id", {}).get("videoId")
            if vid:
                ids.append(vid)

        fetched += len(items)
        token = data.get("nextPageToken")
        if not token:
            break

    return list(dict.fromkeys(ids))

def youtube_videos_stats(video_ids):
    rows = []

    for i in range(0, len(video_ids), 50):
        chunk = ",".join(video_ids[i:i+50])
        data = yt_get("videos", {
            "part": "snippet,statistics",
            "id": chunk
        }, 0.05)

        for it in data.get("items", []):
            s = it["snippet"]
            stt = it.get("statistics", {})
            rows.append({
                "videoId": it["id"],
                "title": s.get("title", ""),
                "description": s.get("description", ""),
                "channelTitle": s.get("channelTitle", ""),
                "publishedAt": s.get("publishedAt", ""),
                "viewCount": int(stt.get("viewCount", 0)),
                "likeCount": int(stt.get("likeCount", 0)),
                "commentCount": int(stt.get("commentCount", 0)),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["publishedAt"] = pd.to_datetime(df["publishedAt"])
        df["ER(%)"] = np.where(
            df["viewCount"] > 0,
            (df["likeCount"] + df["commentCount"]) / df["viewCount"] * 100,
            0
        ).round(3)

    return df

# =============================
# 텍스트 처리
# =============================
def tokenize(text):
    return [
        t for t in TOKEN_PATTERN.findall(text.lower())
        if t not in DEFAULT_STOPWORDS and len(t) >= 2
    ]

def keywords_from_df(df, topn=100):
    corpus = []
    for _, r in df.iterrows():
        corpus.extend(tokenize(f"{r['title']} {r['description']}"))
    return Counter(corpus).most_common(topn)

def draw_wordcloud(freqs):
    wc = WordCloud(
        width=900,
        height=500,
        background_color="white",
        font_path=os.path.join("fonts", "Pretendard-Regular.otf"),
        collocations=False
    )
    wc.generate_from_frequencies(dict(freqs))
    fig = plt.figure(figsize=(9, 5))
    plt.imshow(wc)
    plt.axis("off")
    return fig

# =============================
# UI
# =============================
st.title("📈 유튜브 키워드 트렌드 분석기")
st.caption("키워드/기간 기준으로 영상 성과와 키워드를 분석합니다.")

with st.sidebar:
    keyword = st.text_input("키워드", "브이로그")
    days = st.number_input("최근 N일", 1, 365, 30)
    max_results = st.slider("영상 수", 10, 200, 80, 10)
    region_code = st.text_input("지역 코드", "KR").strip().upper()

    if API_KEY:
        st.success("API Key 로드 완료")
    else:
        st.error("API Key 없음")

    run = st.button("분석 실행", type="primary")

now = datetime.now(timezone.utc)
after = (now - timedelta(days=days)).isoformat()
before = now.isoformat()

if run and API_KEY:
    ids = youtube_search(keyword, after, before, region_code, max_results)
    df = youtube_videos_stats(ids)

    if not df.empty:
        st.success(f"{len(df)}개 영상 분석 완료")

        c1, c2, c3 = st.columns(3)
        c1.metric("총 조회수", f"{df['viewCount'].sum():,}")
        c2.metric("평균 ER(%)", f"{df['ER(%)'].mean():.2f}")
        c3.metric("평균 댓글 수", f"{df['commentCount'].mean():.1f}")

        st.subheader("상위 영상")
        st.dataframe(df.sort_values("ER(%)", ascending=False), use_container_width=True)

        st.subheader("워드클라우드")
        st.pyplot(draw_wordcloud(keywords_from_df(df)), clear_figure=True)

        st.download_button(
            "CSV 다운로드",
            df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"yt_{keyword}_{days}d.csv",
            mime="text/csv"
        )
