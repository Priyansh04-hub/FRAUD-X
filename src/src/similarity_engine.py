import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityEngine:
    def __init__(self):
        self.matrix = None

    def fit(self, matrix):
        self.matrix = matrix
        return self

    def find_similar(
        self,
        query_vector,
        top_n=5
    ):
        if self.matrix is None:
            raise ValueError("Similarity engine has not been fitted.")

        similarities = cosine_similarity(
            query_vector,
            self.matrix
        )[0]

        indices = np.argsort(
            similarities
        )[::-1][:top_n]

        return [
            {
                "index": int(index),
                "similarity": float(similarities[index])
            }
            for index in indices
        ]