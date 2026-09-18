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


if __name__ == "__main__":
    unittest.main()
