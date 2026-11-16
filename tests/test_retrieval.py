import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
from retrieval import BM25Index, HashDenseIndex, HybridIndex, RetrievalRouter


def chunk(cid, text, chapter="chapter_I"):
    return {"id": cid, "source_id": "doc", "source": "doc.pdf", "page": 1,
            "text": text, "metadata": {"chapter": chapter}}


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [chunk("a", "giấy phép đào tạo lái xe"),
                       chunk("b", "diện tích trung tâm sát hạch", "chapter_III")]

    def test_all_strategies_return_ranked_unique_results(self):
        for index in (BM25Index(self.chunks), HashDenseIndex(self.chunks), HybridIndex(self.chunks)):
            rows = index.search("giấy phép lái xe", top_k=2)
            self.assertEqual(len({row["id"] for row in rows}), len(rows))
            self.assertEqual([row["rank"] for row in rows], list(range(1, len(rows) + 1)))

    def test_filter_is_user_supplied(self):
        router = RetrievalRouter(self.chunks)
        rows = router.search("trung tâm", filters={"chapter": "chapter_III"}, strategy="hybrid")
        self.assertEqual([row["id"] for row in rows], ["b"])

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValueError):
            RetrievalRouter(self.chunks).search("x", strategy="unknown")


if __name__ == "__main__":
    unittest.main()
