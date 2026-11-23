"""
rag_pipeline.py

This file is the "glue" that connects every piece of our project together.

It has TWO main jobs:

1. PROCESSING a PDF:
   PDF -> extract text -> split into chunks -> create embeddings
   -> build FAISS index -> save to disk

2. ANSWERING a question:
   question -> embed question -> search FAISS for top matching chunks
   -> build a prompt -> send to Gemini -> return the answer

WHY DOES THIS FILE EXIST SEPARATELY FROM app.py?
---------------------------------------------------
app.py is only responsible for the UI (what the user sees and clicks).
This file (rag_pipeline.py) is responsible for the LOGIC (what actually
happens behind the scenes). Keeping them separate means:
- app.py stays small, clean, and focused on Streamlit code.
- rag_pipeline.py can be tested or reused without needing a UI at all.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

from utils.pdf_loader import extract_text_from_pdf, is_pdf_readable
from utils.text_splitter import split_text_into_chunks
from utils.embeddings import create_embeddings, create_single_embedding
from utils.vector_database import (
    build_faiss_index,
    save_vector_store,
    load_vector_store,
    search_similar_chunks,
)

# Load environment variables from the .env file (like our Gemini API key for local dev).
load_dotenv()


def get_gemini_api_key():
    """
    Retrieve the Gemini API key from Streamlit secrets (st.secrets) if available,
    otherwise fall back to environment variables / .env.
    """
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")


def configure_gemini():
    """
    Set up the Gemini API using our API key.

    Raises:
        ValueError if no API key was found.
    """
    api_key = get_gemini_api_key()

    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "Gemini API key is missing. Please add GEMINI_API_KEY to your Streamlit Secrets or .env file."
        )

    genai.configure(api_key=api_key)


def process_pdf(uploaded_file):
    """
    Run the full PDF processing pipeline:
    extract text -> chunk -> embed -> build FAISS index -> save.

    Parameters:
        uploaded_file: the file uploaded through Streamlit

    Returns:
        (success, message) where success is True/False and message
        explains what happened (used to show the user feedback).
    """
    # Step 1: Extract raw text from the PDF.
    extracted_text = extract_text_from_pdf(uploaded_file)

    # Step 2: Check the PDF actually had readable text.
    if not is_pdf_readable(extracted_text):
        return False, "This PDF appears to be blank, scanned, or corrupted. Please try a different file."

    # Step 3: Split the text into overlapping chunks.
    text_chunks = split_text_into_chunks(extracted_text, chunk_size=1000, chunk_overlap=200)

    if len(text_chunks) == 0:
        return False, "Could not split the PDF into readable chunks. Please try a different file."

    # Step 4: Convert every chunk into an embedding vector.
    chunk_embeddings = create_embeddings(text_chunks)

    # Step 5: Build a FAISS index from those embeddings.
    faiss_index = build_faiss_index(chunk_embeddings)

    # Step 6: Save the index AND the original text chunks to disk,
    # so we don't have to redo this work next time.
    save_vector_store(faiss_index, text_chunks)

    return True, f"PDF processed successfully! Created {len(text_chunks)} chunks."


def answer_question(user_question):
    """
    Run the full retrieval + generation pipeline for a single question.

    Parameters:
        user_question: the question typed by the user (string)

    Returns:
        A string containing Gemini's answer (or a friendly error message).
    """
    # Basic validation: don't process empty questions.
    if not user_question or user_question.strip() == "":
        return "Please type a question before submitting."

    # Step 1: Load the previously saved FAISS index and text chunks.
    faiss_index, text_chunks = load_vector_store()

    if faiss_index is None:
        return "No document has been processed yet. Please upload and process a PDF first."

    # Step 2: Convert the user's question into an embedding vector,
    # using the SAME embedding model we used for the PDF chunks.
    # This is essential — questions and chunks must live in the
    # same "meaning space" to be compared.
    question_embedding = create_single_embedding(user_question)

    # Step 3: Search FAISS for the top 3 most similar chunks.
    top_chunks = search_similar_chunks(faiss_index, text_chunks, question_embedding, top_k=3)

    if len(top_chunks) == 0:
        return "I couldn't find that information in the uploaded document."

    # Step 4: Build the context by joining the top chunks together.
    context_text = "\n\n---\n\n".join(top_chunks)

    # Step 5: Build our prompt. This is where PROMPT ENGINEERING matters:
    # we are explicitly instructing Gemini to ONLY use the given context,
    # and to admit when it doesn't know, instead of guessing. This one
    # instruction is what prevents hallucination in our chatbot.
    prompt = f"""You are a helpful assistant.

Answer ONLY from the provided context.
If the answer is not found in the context, say:
"I couldn't find that information in the uploaded document."

Context:
{context_text}

Question:
{user_question}

Answer:"""

    # Step 6: Send the prompt to Gemini and get the answer back.
    try:
        configure_gemini()
        model = genai.GenerativeModel("gemini-3.6-flash")
        response = model.generate_content(prompt)
        return response.text.strip()

    except ValueError as api_key_error:
        # This catches our own "missing API key" error from configure_gemini().
        return str(api_key_error)

    except Exception as e:
        # This catches any Gemini API failure (bad key, network issue,
        # rate limit, etc.) and shows a friendly message instead of
        # crashing the app.
        return f"Sorry, something went wrong while contacting Gemini: {str(e)}"
