"""
vector_database.py

This file is responsible for ONE job only:
Storing embeddings so we can search them quickly, using FAISS.

WHAT IS A VECTOR DATABASE?
-----------------------------
Once we've converted all our PDF chunks into embeddings (lists of numbers),
we need a way to store them and quickly find "which stored vector is most
similar to this new vector?" A vector database is built exactly for this
job — searching through thousands (or millions) of vectors very fast.

WHAT IS FAISS?
-----------------------------
FAISS (Facebook AI Similarity Search) is a free, open-source library made
by Meta AI for exactly this purpose: fast similarity search over vectors.
It runs entirely on your local machine — no server, no cloud account,
no internet connection needed.

HOW DOES NEAREST-NEIGHBOR SEARCH WORK?
-----------------------------
Imagine each embedding as a point in space (just like a dot on a map,
except with many more than 2 dimensions — our model produces 384 numbers
per chunk!). When we search, FAISS measures the DISTANCE between the
question's point and every stored chunk's point, then returns the chunks
whose points are closest. "Closest" here roughly means "most similar
in meaning."

We use "L2 distance" (straight-line distance) in this project because
it's simple to understand and works well for our beginner use case.

WHY SAVE THE INDEX TO DISK?
-----------------------------
Creating embeddings takes time. If we rebuild them every single time the
app runs, uploading the same PDF twice would be wasteful. By saving the
FAISS index (and the matching chunks) to disk, we can reload everything
instantly next time, without redoing the work.
"""

import os
import pickle
import faiss
import numpy as np

# Folder where we save our FAISS index and the text chunks that go with it.
VECTOR_STORE_DIR = "vector_store"
INDEX_FILE_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index.bin")
CHUNKS_FILE_PATH = os.path.join(VECTOR_STORE_DIR, "text_chunks.pkl")


def build_faiss_index(embeddings):
    """
    Build a brand-new FAISS index from a set of embeddings.

    Parameters:
        embeddings: a NumPy array of embedding vectors

    Returns:
        A FAISS index object, ready to be searched.
    """
    # Every embedding vector has the same length (number of dimensions).
    # For all-MiniLM-L6-v2, this is 384 numbers per vector.
    vector_dimension = embeddings.shape[1]

    # IndexFlatL2 is the simplest FAISS index type:
    # it compares a query vector against EVERY stored vector using
    # L2 (straight-line) distance. For a single PDF's worth of chunks,
    # this "brute force" approach is fast enough and very easy to understand.
    index = faiss.IndexFlatL2(vector_dimension)

    # FAISS requires float32 numbers specifically, so we convert here.
    embeddings = np.array(embeddings).astype("float32")

    # Add all our chunk embeddings into the index at once.
    index.add(embeddings)

    return index


def save_vector_store(index, text_chunks):
    """
    Save the FAISS index and the matching text chunks to disk.

    Parameters:
        index: the FAISS index object
        text_chunks: the list of text chunks the index was built from
    """
    # Make sure the vector_store folder exists.
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    # faiss.write_index() saves the index itself (the embeddings + search structure).
    faiss.write_index(index, INDEX_FILE_PATH)

    # We also need to save the ORIGINAL TEXT of each chunk.
    # FAISS only stores numbers — it doesn't remember what text
    # those numbers came from. So we save the chunks separately,
    # in the SAME ORDER they were added to the index. This way,
    # "result number 3 from FAISS" always maps to "text_chunks[3]".
    with open(CHUNKS_FILE_PATH, "wb") as f:
        pickle.dump(text_chunks, f)


def load_vector_store():
    """
    Load a previously saved FAISS index and its text chunks from disk.

    Returns:
        (index, text_chunks) if a saved vector store exists,
        otherwise (None, None).
    """
    if not os.path.exists(INDEX_FILE_PATH) or not os.path.exists(CHUNKS_FILE_PATH):
        return None, None

    index = faiss.read_index(INDEX_FILE_PATH)

    with open(CHUNKS_FILE_PATH, "rb") as f:
        text_chunks = pickle.load(f)

    return index, text_chunks


def search_similar_chunks(index, text_chunks, question_embedding, top_k=3):
    """
    Find the most similar text chunks to a given question embedding.

    Parameters:
        index: the FAISS index to search
        text_chunks: the list of text chunks matching the index
        question_embedding: the embedding vector of the user's question
        top_k: how many top matches to return (default 3)

    Returns:
        A list of the top_k most relevant text chunks (strings).
    """
    # FAISS expects a 2D array (a batch of queries), even if we only
    # have ONE question. So we reshape our single vector into a
    # "batch of 1" and convert it to float32, same as before.
    question_vector = np.array([question_embedding]).astype("float32")

    # index.search() returns two things:
    #   distances -> how far each match is (lower = more similar)
    #   indices   -> WHICH chunks matched, as position numbers
    distances, indices = index.search(question_vector, top_k)

    # indices[0] gives us the top_k positions for our one question.
    # We use these positions to look up the actual text from text_chunks.
    matching_chunks = []
    for position in indices[0]:
        # FAISS can return -1 if there aren't enough chunks to fill top_k.
        if position == -1:
            continue
        matching_chunks.append(text_chunks[position])

    return matching_chunks
