import streamlit as st
import time
import tempfile
from pathlib import Path

st.set_page_config(page_title="FrameWise", layout="wide")

st.markdown("""
<style>
body, .stApp { background: #f7f9fc; color: #111; }
.block-container { padding: 2rem 3rem; }

h1 { font-size: 32px; margin-bottom: 0; }
.sub { color: #666; font-size: 14px; margin-bottom: 24px; }

.card {
    background: #ffffff;
    border: 1px solid #e6e8eb;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.chat-user {
    text-align: right;
    background: #e8f0fe;
    padding: 10px 14px;
    border-radius: 12px;
    margin-bottom: 8px;
}

.chat-ai {
    background: #f1f3f6;
    padding: 10px 14px;
    border-radius: 12px;
    margin-bottom: 8px;
}

button {
    border-radius: 8px !important;
    font-weight: 500 !important;
}

.empty {
    text-align:center;
    padding:20px;
    color:#666;
}
</style>
""", unsafe_allow_html=True)

for k, v in {
    "processed": False,
    "processing": False,
    "summary": None,
    "chat": [],
    "video_path": None,
    "video_name": None,
    "seek_time": 0
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def run_pipeline(video_path):
    time.sleep(2)

    class Segment:
        def __init__(self, t, txt):
            self.start = t
            self.text = txt

    class Transcript:
        segments = [
            Segment(0, "Intro to machine learning"),
            Segment(60, "Gradient descent explained"),
            Segment(120, "Overfitting and regularization"),
        ]

    return Transcript()


def generate_summary(transcript):
    return "This video explains machine learning fundamentals including gradient descent, overfitting, and core training concepts."


def answer_question(q):
    q = q.lower()
    if "gradient" in q:
        return "Gradient descent is an optimization algorithm used to minimize loss.", [60, 75, 90]
    if "overfitting" in q:
        return "Overfitting occurs when a model memorizes training data and fails to generalize.", [120, 140]
    return "Relevant content found in video.", [0]


def format_ts(t):
    m = int(t) // 60
    s = int(t) % 60
    return f"{m}:{s:02d}"


st.title("🎞️ FrameWise")
st.markdown('<div class="sub">Video Intelligence · Multimodal RAG</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

# LEFT PANEL
with col1:
    uploaded = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

    if uploaded:
        if st.session_state.video_name != uploaded.name:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
            tmp.write(uploaded.read())
            st.session_state.video_path = tmp.name
            st.session_state.video_name = uploaded.name
            st.session_state.processed = False
            st.session_state.chat = []
            st.session_state.summary = None

        st.markdown(f'<div class="card"><b>{uploaded.name}</b></div>', unsafe_allow_html=True)

    if st.button("Process Video", use_container_width=True):
        if not uploaded:
            st.warning("Upload a video to begin")
        else:
            st.session_state.processing = True
            transcript = run_pipeline(st.session_state.video_path)
            st.session_state.transcript = transcript
            st.session_state.processed = True
            st.session_state.processing = False
            st.rerun()


# RIGHT PANEL
with col2:

    if st.session_state.video_path:
        st.video(st.session_state.video_path, start_time=int(st.session_state.seek_time))

    if st.session_state.processing:
        st.info("Processing video...")

    # SUMMARY
    st.markdown('<div class="card"><b>📋 Summary</b><br><br>', unsafe_allow_html=True)

    if not st.session_state.processed:
        st.markdown("""
        <div class="empty">
            <div style="font-size:28px;">🎬</div>
            <div style="font-weight:500; margin-top:8px;">
                Your video summary will appear here
            </div>
            <div style="font-size:13px; margin-top:6px;">
                Upload and process a video to unlock insights ✨
            </div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("Generate Summary"):
        if not st.session_state.processed:
            st.warning("No video processed yet")
        else:
            st.session_state.summary = generate_summary(st.session_state.transcript)

    if st.session_state.summary:
        st.write(st.session_state.summary)

    st.markdown('</div>', unsafe_allow_html=True)

    # Q&A
    st.markdown('<div class="card"><b>💬 Ask Questions</b><br><br>', unsafe_allow_html=True)

    if not st.session_state.processed:
        st.markdown("""
        <div class="empty">
            <div style="font-size:24px;">🧠</div>
            <div style="font-weight:500; margin-top:6px;">
                Ask anything about your video
            </div>
            <div style="font-size:13px; margin-top:4px;">
                Process a video first to start chatting
            </div>
        </div>
        """, unsafe_allow_html=True)

    for idx, msg in enumerate(st.session_state.chat):
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">{msg["text"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai">{msg["text"]}</div>', unsafe_allow_html=True)

            if msg.get("ts"):
                cols = st.columns(len(msg["ts"]))
                for i, t in enumerate(msg["ts"]):
                    with cols[i]:
                        if st.button(f"⏱ {format_ts(t)}", key=f"{idx}_{i}"):
                            st.session_state.seek_time = t
                            st.rerun()

    q_col, btn_col = st.columns([4, 1])

    with q_col:
        q = st.text_input(
            "",
            placeholder="Ask about scenes, concepts, or moments in the video...",
            disabled=not st.session_state.processed
        )

    with btn_col:
        ask = st.button("Ask", disabled=not st.session_state.processed)

    if ask and q.strip():
        st.session_state.chat.append({"role": "user", "text": q})
        ans, ts = answer_question(q)
        st.session_state.chat.append({"role": "ai", "text": ans, "ts": ts})
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)