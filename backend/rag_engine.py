import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RAGEngine:
    def __init__(self, docs_path="docs"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.docs_path = os.path.join(base_dir, docs_path)
        self.chunks = []
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self.tfidf_matrix = None
        self._load_and_index()

    def _load_and_index(self):
        md_files = glob.glob(os.path.join(self.docs_path, "*.md"))
        for file_path in md_files:
            filename = os.path.basename(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            # 문단(## 또는 빈 줄) 단위로 청킹
            raw_sections = [s.strip() for s in content.split("\n\n") if len(s.strip()) > 20]
            for sec in raw_sections:
                self.chunks.append(f"[{filename}] {sec}")

        if self.chunks:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)

    def search(self, query: str, k: int = 3) -> list[str]:
        if not self.chunks or self.tfidf_matrix is None:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = scores.argsort()[::-1][:k]
        return [self.chunks[i] for i in top_indices if scores[i] > 0.05]
