import unittest
import json
from pathlib import Path

from evals.evaluation_report import _rag_corpus_metrics
from rag.knowledge_base import HealthKnowledgeBase
from rag.source_audit import audit


class RagProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]

    def test_legacy_summaries_are_excluded_and_curated_sources_are_indexed(self):
        documents = HealthKnowledgeBase()._load_documents()
        sources = {doc.metadata["source"] for doc in documents}
        manifest = __import__("json").loads((self.root / "backend" / "rag" / "source_manifest.json").read_text(encoding="utf-8"))
        expected = {source for source, entry in manifest.items() if entry.get("review_status") in {"source_curated", "local_imported"}}
        self.assertEqual(sources, expected)
        self.assertGreaterEqual(len(sources), 6)
        self.assertGreaterEqual(sum(doc.metadata.get("review_status") == "local_imported" for doc in documents), 1)
        self.assertFalse(any(source.startswith("who_") for source in sources))
        required = {"publisher", "source_url", "retrieved_at", "jurisdiction", "document_type", "review_status", "version", "content_hash"}
        self.assertTrue(all(required.issubset(doc.metadata) for doc in documents))

    def test_source_audit_reports_legacy_files_without_indexing_them(self):
        report = audit(self.root)
        manifest = __import__("json").loads((self.root / "backend" / "rag" / "source_manifest.json").read_text(encoding="utf-8"))
        expected_count = sum(entry.get("review_status") in {"source_curated", "local_imported"} for entry in manifest.values())
        self.assertEqual(report["active_source_count"], expected_count)
        excluded = {row["source"] for row in report["excluded_sources"]}
        self.assertTrue({"medical_knowledge", "pharmacy_knowledge", "insurance_policies", "lab_reference"}.issubset(excluded))

    def test_corpus_metrics_count_cleaned_active_sources(self):
        metrics = _rag_corpus_metrics(self.root)
        self.assertEqual(metrics["document_count"], 390)
        self.assertEqual(metrics["active_source_count"], 390)
        self.assertGreater(metrics["chunk_count"], 0)

    def test_cleaned_pdfs_exclude_page_numbers_and_reference_tail(self):
        cleaned = self.root / "backend" / "rag" / "sources" / "cleaned"
        for path in cleaned.glob("cn_*.md"):
            text = path.read_text(encoding="utf-8")
            self.assertNotRegex(text, r"(?m)^\[第\d+页\]$")
        obesity = (cleaned / "cn_obesity_guideline_2024.md").read_text(encoding="utf-8")
        self.assertNotIn("## 1975.DOI", obesity)
        self.assertNotIn("10.1056/", obesity)


if __name__ == "__main__":
    unittest.main()
