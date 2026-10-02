import re
from sklearn.feature_extraction.text import TfidfVectorizer

class TransactionTextProcessor:
    def __init__(self, max_features=100):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english"
        )

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def prepare_text(self, values):
        return [
            self.clean_text(value)
            for value in values
        ]

    def fit_transform(self, values):
        cleaned = self.prepare_text(values)
        return self.vectorizer.fit_transform(cleaned)

    def transform(self, values):
        cleaned = self.prepare_text(values)
        return self.vectorizer.transform(cleaned)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()