import streamlit as st
import time
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Importing your specific backend classes
from framewise import (
    TranscriptExtractor,
    FrameExtractor,
    FrameWiseEmbedder,
    FrameWiseVectorStore,
    FrameWiseQA,
)

load_dotenv()

# --- Page Config & Custom Styling (Preserved from your previous design) ---
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

# --- Session State Management ---
# Initializing keys if they don't exist
if "processed" not in st.session_state:
    st.session_state.update({
        "processed": False,
        "processing": False,
        "summary": None,
        "chat": [],
        "video_path": None,
        "video_name": None,
        "seek_time": 0,
        "transcript_obj": None,
        "qa_engine": None
    })

def format_ts(t):
    m = int(t) // 60
    s = int(t) % 60
    return f"{m}:{s:02d}"

# --- Title Header ---
st.title("🎞️ FrameWise")
st.markdown('<div class="sub">Video Intelligence · Multimodal RAG</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

# --- LEFT PANEL: Upload & Process ---
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
            
            # Real Backend Pipeline Execution
            with st.spinner("Analyzing Video Content..."):
                # 1. Extract Transcript
                te = TranscriptExtractor(model_size="base")
                transcript = te.extract(st.session_state.video_path)
                st.session_state.transcript_obj = transcript
                
                # 2. Extract & Embed Frames (Hybrid Strategy)
                fe = FrameExtractor(strategy="hybrid", max_frames_per_video=15)
                frames_dir = Path("app_data/frames")
                frames = fe.extract(st.session_state.video_path, transcript, output_dir=str(frames_dir))
                
                embedder = FrameWiseEmbedder(device="cpu")
                embeddings = embedder.embed_frames_batch(frames)
                
                # 3. Setup Vector Store & QA Engine
                vector_store = FrameWiseVectorStore(db_path="app_data/framewise.db")
                vector_store.create_table(embeddings, mode="overwrite")
                
                st.session_state.qa_engine = FrameWiseQA(
                    vector_store=vector_store,
                    embedder=embedder,
                    model="gemini-1.5-flash" # Ensure this matches your package capability
                )
                
                st.session_state.processed = True
                st.session_state.processing = False
                st.rerun()

# --- RIGHT PANEL: Video, Summary & Chat ---
with col2:
    if st.session_state.video_path:
        st.video(st.session_state.video_path, start_time=int(st.session_state.seek_time))

    if st.session_state.processing:
        st.info("Processing video with Whisper and CLIP...")

    # --- SUMMARY SECTION ---
    st.markdown('<div class="card"><b>📋 Summary</b><br><br>', unsafe_allow_html=True)

    if not st.session_state.processed:
        st.markdown("""
        <div class="empty">
            <div style="font-size:28px;">🎬</div>
            <div style="font-weight:500; margin-top:8px;">Your video summary will appear here</div>
            <div style="font-size:13px; margin-top:6px;">Upload and process a video to unlock insights ✨</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Check if summary already exists in state
        if st.session_state.summary:
            st.write(st.session_state.summary)
        else:
            if st.button("Generate Summary"):
                # Using the full_text from your Transcript class to generate summary via Gemini
                prompt = f"Provide a concise summary of this video based on the transcript: {st.session_state.transcript_obj.full_text[:4000]}"
                response = st.session_state.qa_engine.ask(prompt)
                st.session_state.summary = response['answer']
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Q&A SECTION ---
    st.markdown('<div class="card"><b>💬 Ask Questions</b><br><br>', unsafe_allow_html=True)

    if not st.session_state.processed:
        st.markdown("""
        <div class="empty">
            <div style="font-size:24px;">🧠</div>
            <div style="font-weight:500; margin-top:6px;">Ask anything about your video</div>
            <div style="font-size:13px; margin-top:4px;">Process a video first to start chatting</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Display Chat History
        for idx, msg in enumerate(st.session_state.chat):
            role_css = "chat-user" if msg["role"] == "user" else "chat-ai"
            st.markdown(f'<div class="{role_css}">{msg["text"]}</div>', unsafe_allow_html=True)

            if msg.get("ts"):
                # Create timestamp buttons for navigation
                ts_count = len(msg["ts"])
                if ts_count > 0:
                    ts_cols = st.columns(min(ts_count, 4))
                    for i, t in enumerate(msg["ts"][:4]): # Showing up to 4 timestamps
                        with ts_cols[i]:
                            if st.button(f"⏱ {format_ts(t)}", key=f"chat_ts_{idx}_{i}"):
                                st.session_state.seek_time = t
                                st.rerun()

        # Input Area
        q_col, btn_col = st.columns([4, 1])
        with q_col:
            user_query = st.text_input(
                "",
                placeholder="Ask about scenes, concepts, or moments...",
                key="query_input",
                disabled=not st.session_state.processed
            )

        with btn_col:
            ask_clicked = st.button("Ask", disabled=not st.session_state.processed)

        if ask_clicked and user_query.strip():
            st.session_state.chat.append({"role": "user", "text": user_query})
            
            # Fetch real response from QA Engine
            response = st.session_state.qa_engine.ask(user_query, num_results=3)
            
            # Extract timestamps from the relevant_frames metadata
            timestamps = [f['timestamp'] for f in response.get('relevant_frames', [])]
            
            st.session_state.chat.append({
                "role": "ai", 
                "text": response['answer'], 
                "ts": timestamps
            })
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)