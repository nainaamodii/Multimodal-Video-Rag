import streamlit as st
import os
from pathlib import Path
from loguru import logger

# Import your existing logic (assumes your script is named final_pipeline.py or similar)
# If the code you provided is in test_qa.py, import from there.
from test.test_qa import (
    derive_video_meta, 
    is_already_processed, 
    process_video, 
    load_existing, 
    build_qa, 
    get_summary
)

# --- Page Config ---
st.set_page_config(page_title="EduQuery - Video AI", layout="wide")

# --- Styling ---
st.markdown("""
    <style>
    .stChatFloatingInputContainer { bottom: 20px; }
    .main { background-color: #f5f7f9; }
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "processed_video_id" not in st.session_state:
    st.session_state.processed_video_id = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Resource Caching ---
@st.cache_resource
def get_pipeline(video_path_str, force_reprocess=False):
    """Handles the heavy lifting of ingestion and model loading."""
    video_path = Path(video_path_str)
    meta = derive_video_meta(video_path)
    
    if force_reprocess or not is_already_processed(meta):
        vector_store, embedder = process_video(video_path, meta)
    else:
        vector_store, embedder = load_existing(meta)
    
    qa_system = build_qa(vector_store, embedder)
    return qa_system, meta

# --- Main Logic ---
def main():
    st.title("🎓 EduQuery: Video Learning Assistant")
    
    # 1. File Uploader
    with st.sidebar:
        st.header("Upload Lecture")
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
        
        if uploaded_file:
            # Save file locally to process it
            temp_dir = Path("temp_uploads")
            temp_dir.mkdir(exist_ok=True)
            video_path = temp_dir / uploaded_file.name
            
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("🚀 Process Video") or st.session_state.processed_video_id != uploaded_file.name:
                with st.status("Analyzing video... this may take a minute", expanded=True) as status:
                    qa_system, meta = get_pipeline(str(video_path))
                    st.session_state.summary = get_summary(qa_system)
                    st.session_state.qa_system = qa_system
                    st.session_state.meta = meta
                    st.session_state.processed_video_id = uploaded_file.name
                    st.session_state.messages = [] # Reset chat for new video
                    status.update(label="Analysis Complete!", state="complete")
                st.rerun()

        if st.session_state.processed_video_id:
            if st.button("🧹 Clear & Process New Video"):
                st.session_state.clear()
                st.rerun()

    # 2. UI Layout
    if st.session_state.processed_video_id:
        col1, col2 = st.columns([1, 1], gap="large")

        # LEFT COLUMN: Video & Summary
        with col1:
            st.subheader("📺 Lecture View")
            # Using the original file path from meta
            st.video(str(Path("temp_uploads") / st.session_state.processed_video_id))
            
            st.divider()
            
            st.subheader("📝 AI Summary")
            st.markdown(st.session_state.summary)

        # RIGHT COLUMN: QA Chat
        with col2:
            st.subheader("💬 Ask Questions")
            chat_container = st.container(height=500)
            
            # Display chat history
            for message in st.session_state.messages:
                with chat_container.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input("What is gradient descent?"):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with chat_container.chat_message("user"):
                    st.markdown(prompt)

                # Generate response
                with chat_container.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = st.session_state.qa_system.ask(prompt, top_k=5)
                        answer = response["answer"]
                        st.markdown(answer)
                        
                        # Show sources in an expander
                        if response.get("sources"):
                            with st.expander("View Sources from Video"):
                                for src in response["sources"]:
                                    t = src['start_time']
                                    st.caption(f"📍 At {int(t//60)}:{int(t%60):02d} — {src.get('text_preview', '')[:100]}...")

                st.session_state.messages.append({"role": "assistant", "content": answer})

    else:
        st.info("Please upload and process a video file from the sidebar to begin.")

if __name__ == "__main__":
    main()