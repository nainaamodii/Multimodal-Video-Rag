# 🎓 EduQuery — Multimodal Video RAG

A Retrieval-Augmented Generation (RAG) system for educational video content. Upload a lecture video, get an AI-generated summary, and ask natural-language questions — with timestamped source citations pulled directly from the video.

## ✨ Features

- 🎬 **Video ingestion** — upload MP4, MOV, or AVI lecture files
- 🗣️ **Audio transcription** — powered by OpenAI Whisper
- 🖼️ **Visual understanding** — frame-level analysis via computer vision
- 🔍 **Semantic search** — embeddings stored in LanceDB vector store
- 💬 **Conversational Q&A** — ask questions, get answers with video timestamp sources
- 📝 **Auto-summary** — AI-generated lecture overview on ingestion
- ⚡ **Smart caching** — already-processed videos are loaded instantly

## 🗂️ Project Structure
```
Multimodal-Video-Rag/
├── api/                  # API layer
├── dataset/              # Sample/test datasets
├── ingestion/            # Video processing & embedding pipeline
├── processed_videos/     # Cache of processed video data
├── query/                # RAG query & retrieval logic
├── storage/              # Vector store interfaces (LanceDB)
├── test/                 # Tests and QA pipeline (test_qa.py)
├── ui/                   # UI components
├── main.py               # Streamlit app entrypoint
├── pyproject.toml        # Project metadata & dependencies
└── req.txt               # pip requirements
```

## 🚀 Getting Started

### Prerequisites

- Python ≥ 3.11
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation
```bash
# Clone the repo
git clone https://github.com/nainaamodii/Multimodal-Video-Rag.git
cd Multimodal-Video-Rag

# Install with uv (recommended)
uv sync

# Or with pip
pip install -r req.txt
```

### Environment Variables

Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_google_gemini_api_key
OPENAI_API_KEY=your_openai_api_key   # optional, if using OpenAI models
```

### Run the App
```bash
streamlit run main.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

## 🧠 How It Works

1. **Upload** a lecture video via the sidebar
2. **Ingestion pipeline** extracts audio (Whisper transcription) and video frames (OpenCV), chunks and embeds everything using `sentence-transformers`, and stores vectors in LanceDB
3. **Ask questions** — your query is embedded and matched against stored chunks; a Gemini/LLM model generates an answer grounded in the retrieved segments
4. **Sources** — each answer links back to the exact timestamp in the video

## 🛠️ Tech Stack

| Component | Library |
|---|---|
| UI | Streamlit |
| Transcription | OpenAI Whisper |
| Embeddings | sentence-transformers |
| Vector Store | LanceDB |
| LLM | Google Gemini (`langchain-google-genai`) |
| Video Processing | OpenCV, Pillow |
| Deep Learning | PyTorch, torchvision |
| Logging | Loguru |

## 📦 Key Dependencies
```toml
openai >= 1.0.0
transformers >= 4.0.0
sentence-transformers >= 2.0.0
lancedb >= 0.5.0
opencv-python >= 4.0.0
torch >= 2.0.0
langchain-google-genai >= 0.1.0
openai-whisper >= 20231117
streamlit >= 1.56.0
```

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

### Built with ❤️ by Team 
