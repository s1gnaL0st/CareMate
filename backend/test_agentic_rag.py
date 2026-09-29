import unittest
from dataclasses import dataclass

from agentic_rag import retrieve_until_sufficient


@dataclass
class Doc:
    page_content: str
    metadata: dict


class FakeRetriever:
    def __init__(self):
        self.queries = []

    async def aretrieve(self, query: str, k: int = 3):
        self.queries.append(query)
        if len(self.queries) == 1:
            return [Doc("药物适应症和用法", {"parent_id": "drug::indication"})]
        return [
            Doc("药物适应症和用法", {"parent_id": "drug::indication"}),
            Doc("禁忌 不良反应 相互作用 合并用药", {"parent_id": "drug::safety"}),
        ]


class AgenticRagTests(unittest.IsolatedAsyncioTestCase):
    async def test_controller_retrieves_again_after_observing_missing_facet(self):
        retriever = FakeRetriever()
        result = await retrieve_until_sufficient(
            "这个药能不能吃", retriever, k=2, max_rounds=3, min_sources=2
        )
        self.assertTrue(result.sufficient)
        self.assertEqual(len(retriever.queries), 2)
        self.assertEqual(len(result.rounds), 2)
        self.assertEqual(result.rounds[0].missing_facets, ["safety"])
        self.assertEqual(result.stop_reason, "evidence_sufficient")

    async def test_controller_reports_insufficient_evidence_after_budget(self):
        class Empty:
            async def aretrieve(self, query, k=3):
                return []

        result = await retrieve_until_sufficient("这个药能不能吃", Empty(), max_rounds=2)
        self.assertFalse(result.sufficient)
        self.assertEqual(len(result.rounds), 2)
        self.assertIn("indication", result.missing_facets)
        self.assertEqual(result.stop_reason, "max_rounds")


if __name__ == "__main__":
    unittest.main()
