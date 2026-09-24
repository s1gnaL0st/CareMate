import unittest

from rag.knowledge_base import HealthKnowledgeBase


class RagMetadataTests(unittest.TestCase):
    def test_source_documents_include_traceable_metadata(self):
        documents = HealthKnowledgeBase()._load_documents()
        self.assertTrue(documents)
        for document in documents:
            self.assertTrue(document.metadata.get("source"))
            self.assertTrue(document.metadata.get("section"))
            self.assertTrue(document.metadata.get("updated_at"))

    def test_index_documents_are_child_chunks_with_parent_identity(self):
        documents = HealthKnowledgeBase()._load_documents()
        self.assertTrue(documents)
        self.assertTrue(all(document.metadata.get("chunk_type") == "child" for document in documents))
        self.assertTrue(all(document.metadata.get("parent_id") for document in documents))

    def test_retrieval_returns_unique_parent_sections(self):
        kb = HealthKnowledgeBase()
        results = kb.retrieve("高血压筛查和诊断", k=5)
        self.assertTrue(results)
        parent_ids = [document.metadata.get("parent_id") for document in results]
        self.assertEqual(len(parent_ids), len(set(parent_ids)))
        self.assertTrue(all(document.metadata.get("chunk_type") == "parent" for document in results))
        self.assertTrue(all(document.metadata.get("matched_child") for document in results))


if __name__ == "__main__":
    unittest.main()
