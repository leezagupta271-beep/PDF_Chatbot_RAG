# 📄 PDF RAG Chatbot

A simple, beginner-friendly chatbot that answers questions **only** based on
the content of a PDF you upload — built using Streamlit, FAISS, Sentence
Transformers, and Google Gemini.

This project is designed as a **learning project** for understanding
Retrieval-Augmented Generation (RAG), not as a production-ready system.

---

## 🧠 Project Overview

You upload a PDF. The app:

1. Extracts the text from the PDF (`PyPDF2`)
2. Splits the text into overlapping chunks
3. Converts each chunk into an embedding vector (`sentence-transformers`)
4. Stores those vectors in a local FAISS index
5. When you ask a question, it embeds your question, finds the top 3
   most relevant chunks, and sends **only those chunks** to Gemini
6. Gemini answers strictly from that context — if the answer isn't in
   the PDF, it says so instead of guessing

---

## 📂 Folder Structure

```
PDF_RAG_CHATBOT/
│
├── app.py                     # Streamlit UI
├── requirements.txt           # Python dependencies
├── .env                       # Your Gemini API key (keep secret!)
├── README.md                  # This file
│
├── data/                      # (optional) place to keep sample PDFs
├── vector_store/              # FAISS index + chunks get saved here
│
├── utils/
│   ├── pdf_loader.py          # Extracts text from PDF
│   ├── text_splitter.py       # Splits text into overlapping chunks
│   ├── embeddings.py          # Converts text into vectors
│   ├── vector_database.py     # FAISS storage + similarity search
│   └── rag_pipeline.py        # Connects retrieval + Gemini generation
│
└── assets/                    # (optional) images/logo for the UI
```

---

## ⚙️ Installation

### 1. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### 2. Install requirements

```bash
pip install -r requirements.txt
```

---

## 🔑 How to Get a Gemini API Key

1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **Create API Key**
4. Copy the key and paste it into your `.env` file:

```
GEMINI_API_KEY=your_actual_key_here
```

**Never share this key or commit `.env` to GitHub.** Anyone with your
key can use your Gemini quota.

---

## ▶️ How to Run

```bash
streamlit run app.py
```

This opens the app in your browser (usually at `http://localhost:8501`).

**Steps in the app:**
1. Upload a PDF in the sidebar
2. Click **Process PDF** and wait for the success message
3. Type a question in the chat box at the bottom
4. Read the answer — it comes only from your PDF!

---

## 💬 Example Questions

If you upload a resume PDF, try:
- "What is this person's most recent job title?"
- "List their technical skills."
- "Do they have any certifications?"

If you upload a research paper, try:
- "What is the main conclusion of this paper?"
- "What dataset was used?"
- "Summarize the methodology in 2 sentences."

If you ask something NOT in the PDF, you should see:
> "I couldn't find that information in the uploaded document."

---

## 🧩 How RAG Works (Quick Recap)

**RAG = Retrieval + Generation**

Instead of asking Gemini to answer from its general training data (which
can lead to hallucinated/made-up answers), we:

1. **Retrieve** the most relevant pieces of YOUR document using vector
   similarity search (FAISS)
2. **Generate** an answer using Gemini, but restrict it to only use the
   retrieved pieces of text

This keeps answers grounded, accurate, and specific to your document —
instead of vague or made-up general knowledge.

**Key concepts:**
- **Chunking**: breaking a big document into smaller pieces so each piece
  can be embedded (and searched) more precisely
- **Overlap**: repeating a bit of text between chunks so ideas that span
  a chunk boundary aren't lost
- **Embeddings**: turning text into numbers that capture meaning, so
  similar meanings end up as similar numbers
- **FAISS**: a fast local library for finding which stored embeddings are
  closest (most similar) to a new one
- **Prompt engineering**: carefully instructing the model to only answer
  from given context, which is what prevents hallucination here

---

## 🚀 Future Improvements

- Support multiple PDFs at once (merge multiple documents into one index)
- Show which chunk/page an answer came from (source citations)
- Add a "confidence score" based on FAISS distance
- Support scanned/image PDFs using OCR
- Swap `IndexFlatL2` for a faster FAISS index type for very large PDFs
- Add persistent chat history across sessions (currently resets on refresh)
- Pin exact dependency versions in `requirements.txt` for reproducibility

---

## 📝 Notes

- This app currently supports **one PDF at a time**. Processing a new
  PDF overwrites the previous FAISS index in `vector_store/`.
- Everything runs **locally** except the final call to the Gemini API.
- Built for learning — code favors clarity over performance.
