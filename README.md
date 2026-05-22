# AKGEC Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) chatbot designed for **Ajay Kumar Garg Engineering College (AKGEC)**. The chatbot provides intelligent responses based on college-specific data scraped from the official website.

## 🚀 Features

- **RAG Architecture**: Combines LLM reasoning with real-time context retrieval from a vector database.
- **Context-Aware Conversations**: Summarizes previous interactions to maintain coherent multi-turn dialogues.
- **Fast Inference**: Powered by **Groq** for high-speed LLM responses.
- **Vector Search**: Uses **Pinecone** for efficient similarity search over college documentation.
- **Rate Limiting**: Built-in protection against API abuse using **SlowAPI**.
- **Modern UI**: Clean, responsive web interface for seamless user interaction.

## 🛠️ Tech Stack

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/)
- **LLM Engine**: [Groq](https://groq.com/) (Models: `gpt-oss-120b`, `kimi-k2-instruct-0905`)
- **Vector Database**: [Pinecone](https://www.pinecone.io/)
- **Data Scraping**: [Crawl4AI](https://crawl4ai.com/)
- **Frontend**: HTML5, Vanilla CSS, JavaScript
- **Deployment**: [Render](https://render.com/)

## 📁 Project Structure

```text
AKGEC_chatbot/
├── data/               # Scraping scripts and data processing notebooks
├── public/             # Frontend assets (HTML, CSS, JS)
├── src/                # Core source code (API and Agent logic)
├── main.py             # Entry point for FastAPI
├── agents.py           # LLM and Vector DB orchestration
├── models.py           # Pydantic data models
├── requirements.txt    # Python dependencies
└── render.yaml         # Deployment configuration for Render
```

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.9+
- Groq API Key
- Pinecone API Key & Index

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd AKGEC_chatbot
   ```

2. **Create a virtual environment**:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate.ps1  # On CMD -  .venv\Scripts\activate 
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Create a `.env` file in the root directory and add your credentials:
   ```env
   GROQ_API_KEY=your_groq_api_key
   PC_API_KEY=your_pinecone_api_key
   ```

### Running the Application

1. **Start the FastAPI server**:
   ```bash
   uvicorn main:app --reload
   ```

2. **Access the Chatbot**:
   - Frontend: `http://localhost:8000/`
   - API Docs (Swagger): `http://localhost:8000/docs`

## 📊 Data Pipeline

The knowledge base is built by scraping the college website and indexing the content:

1. **Scraping**: `data/web-markdown.py` uses `crawl4ai` to extract content from the AKGEC website into Markdown.
2. **Indexing**: `data/Upload_data.ipynb` processes the Markdown files, generates embeddings, and uploads them to a Pinecone index (`akgec-data`).

## 🛡️ API Endpoints

### `POST /api/chat`
Handles user queries and returns AI-generated responses.
- **Rate Limit**: 2 requests/minute, 100 requests/day.
- **Payload**:
  ```json
  {
    "message": "Tell me about the B.Tech CSE department",
    "history": "" 
  }
  ```

## 📝 License
Distributed under the MIT License. See `LICENSE` for more information.
