"""
app.py

This is the FRONT DOOR of our project — the Streamlit user interface.

This file does NOT contain any RAG logic itself. Its only job is to:
1. Show the UI (upload button, chat box, chat history).
2. Call functions from utils/rag_pipeline.py when the user acts.
3. Display the results back to the user.

Keeping UI code separate from logic code (in rag_pipeline.py) is a
good habit: it makes each file easier to understand and change.
"""

import streamlit as st
from utils.rag_pipeline import process_pdf, answer_question

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="wide")

st.title("📄 PDF RAG Chatbot")
st.caption("Upload a PDF, then ask questions. Answers come ONLY from your document.")

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
# Streamlit reruns this entire script every time the user interacts
# with the page. To "remember" things (like chat history) between
# those reruns, we store them in st.session_state, which persists
# for the life of the browser session.

if "chat_history" not in st.session_state:
    # A list of (question, answer) tuples.
    st.session_state.chat_history = []

if "pdf_processed" not in st.session_state:
    # Tracks whether a PDF has been successfully processed yet,
    # so we can warn the user if they try to ask questions too early.
    st.session_state.pdf_processed = False


# ---------------------------------------------------------
# SIDEBAR: Upload + Process + Clear
# ---------------------------------------------------------
with st.sidebar:
    st.header("📁 Document")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    process_button = st.button("Process PDF", use_container_width=True)

    if process_button:
        if uploaded_file is None:
            # Friendly error: no PDF uploaded case.
            st.error("Please upload a PDF file first.")
        else:
            with st.spinner("Reading and indexing your PDF... this may take a moment."):
                success, message = process_pdf(uploaded_file)

            if success:
                st.success(message)
                st.session_state.pdf_processed = True
            else:
                # Friendly error: blank/corrupted PDF cases handled
                # inside process_pdf() and returned as 'message'.
                st.error(message)

    st.divider()

    clear_button = st.button("🗑️ Clear Chat", use_container_width=True)
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.markdown(
        "**How it works:**\n"
        "1. Upload a PDF\n"
        "2. Click Process PDF\n"
        "3. Ask questions below\n"
        "4. Answers come only from the PDF"
    )


# ---------------------------------------------------------
# MAIN SCREEN: Chat Interface
# ---------------------------------------------------------

# Display the existing chat history first, so new messages
# appear below the old ones (like a normal chat app).
for question, answer in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(answer)

# The chat input box sits at the bottom of the page, like a real chat app.
user_question = st.chat_input("Ask a question about your PDF...")

if user_question:
    # Friendly error: warn if no PDF has been processed yet.
    if not st.session_state.pdf_processed:
        st.warning("Please upload and process a PDF before asking questions.")
    else:
        # Show the user's question immediately.
        with st.chat_message("user"):
            st.write(user_question)

        # Show a spinner while we retrieve chunks and call Gemini.
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = answer_question(user_question)
            st.write(answer)

        # Save this exchange into our chat history for future reruns.
        st.session_state.chat_history.append((user_question, answer))
