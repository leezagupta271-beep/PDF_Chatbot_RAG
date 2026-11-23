"""
pdf_loader.py

This file is responsible for ONE job only:
Taking a PDF file and turning it into plain text.

Why PyPDF2?
- It is simple, beginner-friendly, and works fully offline.
- It reads PDF files page by page and pulls out the text layer
  (the actual typed/printed text, not images).

Note: PyPDF2 cannot read text from scanned/image-only PDFs.
For this beginner project, we assume the PDF has real (selectable) text.
"""

from PyPDF2 import PdfReader


def extract_text_from_pdf(uploaded_file):
    """
    Extract all text from an uploaded PDF file.

    Parameters:
        uploaded_file: a file-like object (from Streamlit's file_uploader)

    Returns:
        A single string containing all the text from the PDF.
        Returns an empty string if no text could be extracted.
    """

    # PdfReader opens the PDF and lets us loop through its pages.
    # Streamlit gives us the file already in memory, so we can pass
    # it directly to PdfReader without saving it to disk first.
    pdf_reader = PdfReader(uploaded_file)

    # This will hold all the text from every page, combined together.
    full_text = ""

    # Loop through every page in the PDF one by one.
    for page in pdf_reader.pages:

        # extract_text() pulls the text out of a single page.
        # If a page has no text (e.g. a blank page), this can return None,
        # so we default to an empty string using "or ''".
        page_text = page.extract_text() or ""

        # Remove extra leading/trailing whitespace so we can
        # reliably check if the page was actually blank.
        page_text = page_text.strip()

        # Skip blank pages completely — no point adding empty text.
        if page_text == "":
            continue

        # Add this page's text to our running total.
        # We add two newlines so pages are clearly separated,
        # which helps later when we split text into chunks.
        full_text += page_text + "\n\n"

    return full_text.strip()


def is_pdf_readable(extracted_text):
    """
    Check whether a PDF actually produced usable text.

    Parameters:
        extracted_text: the string returned by extract_text_from_pdf()

    Returns:
        True if there is meaningful text, False if the PDF was
        blank, corrupted, or scanned-image-only.
    """
    # If the text is empty or extremely short, treat it as unreadable.
    # A real PDF page almost always has more than 10 characters of text.
    return extracted_text is not None and len(extracted_text.strip()) > 10
