"""
embeddings.py

This file is responsible for ONE job only:
Turning text into embeddings (lists of numbers that capture meaning).

WHAT IS AN EMBEDDING?
-----------------------
Computers don't understand words the way humans do. They understand numbers.
An embedding is a way of converting a piece of text into a list of numbers
(called a "vector") that represents the MEANING of that text.

Think of it like GPS coordinates, but for meaning instead of location.
Just like Paris and London have coordinates that are "close" on a map
because they're both in Europe, the sentences:

    "The cat sat on the mat"
    "A kitten was resting on the rug"

will have embedding vectors that are "close" to each other, because
they mean similar things — even though they don't share many exact words!

This is called SEMANTIC SIMILARITY — similarity in MEANING, not just
matching keywords.

WHY DOES THIS MATTER FOR RAG?
-------------------------------
When a user asks a question, we don't search for exact word matches.
Instead, we convert the question into an embedding too, and then look
for chunks whose embeddings are "closest" to the question's embedding.
This lets us find relevant information even if the user's wording is
completely different from the PDF's wording.

WHY all-MiniLM-L6-v2?
-------------------------------
This is a small, fast, well-tested embedding model from the
sentence-transformers library. It runs entirely on your own computer
(no internet or API needed), and it's a great balance between speed
and accuracy for a beginner project.
"""

from sentence_transformers import SentenceTransformer

# We load the model ONCE, outside of any function.
# Loading a model is slow, so we don't want to reload it every time
# we need an embedding. This line runs once when the file is imported,
# and the model stays in memory for reuse.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(text_chunks):
    """
    Convert a list of text chunks into a list of embedding vectors.

    Parameters:
        text_chunks: a list of strings (our PDF chunks)

    Returns:
        A NumPy array of embeddings, where each row is the
        embedding vector for the corresponding chunk.
    """
    # encode() does all the heavy lifting: it feeds each chunk through
    # the neural network model and returns a vector (list of numbers)
    # for each one. Every chunk becomes a row of numbers.
    embeddings = embedding_model.encode(text_chunks, show_progress_bar=False)

    return embeddings


def create_single_embedding(text):
    """
    Convert a single piece of text (like a user's question) into
    one embedding vector.

    Parameters:
        text: a single string

    Returns:
        A single embedding vector (NumPy array of numbers).
    """
    # We wrap 'text' in a list because encode() expects a list,
    # then take the first (and only) result back out with [0].
    embedding = embedding_model.encode([text], show_progress_bar=False)[0]

    return embedding
