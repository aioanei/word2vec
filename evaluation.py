import numpy as np


def cosine_similarity(vec_a, vec_b):
    # Cosine similarity between two vectors
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def nearest_neighbors(word, embeddings, word2idx, idx2word, top_n=5):
    # Return the top_n most similar words to the given word.

    idx = word2idx[word]
    target_vec = embeddings[idx]

    # Compute cosine similarity against all embeddings at once
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normalized = embeddings / norms
    scores = normalized @ (target_vec / (np.linalg.norm(target_vec) + 1e-10))

    # Exclude the word itself
    scores[idx] = -np.inf
    top_indices = np.argsort(scores)[::-1][:top_n]
    return [(idx2word[i], float(scores[i])) for i in top_indices]


def evaluate(model_name, embeddings, word2idx, idx2word):
    print(f"\n Evaluation: {model_name}")

    # Pick a sample word that exists in vocabulary
    sample = "king"

    # Nearest neighbors
    print(f"\nNearest neighbors of '{sample}':")
    neighbors = nearest_neighbors(sample, embeddings, word2idx, idx2word, top_n=5)
    for word, score in neighbors:
        print(f"{word:} similarity: {score:.4f}")

    # Cosine similarity for a pair
    pairs = [("communist", "capitalist"), ("one", "two")]
    print("\nCosine similarities:")
    for w1, w2 in pairs:
        sim = cosine_similarity(embeddings[word2idx[w1]], embeddings[word2idx[w2]])
        print(f"{w1} <-> {w2} {sim:.4f}")