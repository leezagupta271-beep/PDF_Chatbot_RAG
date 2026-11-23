"""
text_splitter.py

This file is responsible for ONE job only:
Breaking a big block of text into smaller, overlapping chunks.

WHY DO WE NEED CHUNKING?
-------------------------
Imagine trying to summarize an entire 50-page book using a single sentence.
You'd lose almost all the detail. The same problem happens with embeddings:
if we try to create ONE embedding for an entire PDF, that embedding becomes
too vague to represent any specific detail well.

So instead, we break the PDF into small chunks (e.g. 1000 characters each).
Each chunk gets its OWN embedding, which captures its specific meaning
much more accurately. This means when we search later, we can find the
exact paragraph that answers the question — not just "somewhere in the PDF."

WHY DO WE NEED OVERLAP?
-------------------------
Imagine this sentence gets cut exactly in half between two chunks:

    Chunk 1: "...the total revenue for the year was"
    Chunk 2: "1.2 million dollars, a 20% increase..."

Neither chunk alone tells the full story! The important number "1.2 million"
is disconnected from what it refers to ("total revenue").

Overlap fixes this by letting the END of one chunk repeat at the START of
the next chunk. So with overlap, Chunk 2 might actually look like:

    Chunk 2: "...the total revenue for the year was 1.2 million dollars..."

Now the full idea is captured in at least one chunk. This overlap acts like
a safety net so we never lose meaning at the chunk boundaries.
"""


def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    """
    Split a long string of text into smaller overlapping chunks.

    Parameters:
        text: the full text to split (a string)
        chunk_size: how many characters each chunk should contain
        chunk_overlap: how many characters should repeat between
                       consecutive chunks

    Returns:
        A list of text chunks (list of strings)
    """

    # Safety check: if there's no text, return an empty list.
    if not text or len(text.strip()) == 0:
        return []

    chunks = []

    # 'start' marks where the current chunk begins in the full text.
    start = 0
    text_length = len(text)

    while start < text_length:
        # 'end' marks where the current chunk stops.
        # We take a slice of 'chunk_size' characters from 'start'.
        end = start + chunk_size

        # Grab the chunk of text from start to end.
        # Python slicing automatically handles the case where
        # 'end' goes past the end of the text.
        chunk = text[start:end]

        chunks.append(chunk.strip())

        # Move the start position forward for the NEXT chunk.
        # Instead of moving a full chunk_size forward, we move
        # (chunk_size - chunk_overlap) forward. This is what
        # creates the overlapping region between chunks.
        start += (chunk_size - chunk_overlap)

    # Remove any empty chunks that might have been created
    # (this can happen if the text length lines up exactly at the end).
    chunks = [c for c in chunks if len(c) > 0]

    return chunks
