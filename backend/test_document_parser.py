import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from document_parser import DocumentParseError, _normalize_pages, parse_document


class DocumentParserTests(unittest.TestCase):
    def test_normalizes_page_and_block_contract(self):
        pages = _normalize_pages([
            {
                "page": 2,
                "text": "报告正文",
                "blocks": [
                    {"type": "table", "content": "血红蛋白 120", "bbox": [1, 2, 3, 4], "confidence": "0.91"},
                    {"text": "", "confidence": "bad"},
                ],
            }
        ])

        self.assertEqual(pages[0]["page_number"], 2)
        self.assertEqual(pages[0]["blocks"][0]["text"], "血红蛋白 120")
        self.assertEqual(pages[0]["blocks"][0]["confidence"], 0.91)
        self.assertIsNone(pages[0]["blocks"][1]["confidence"])

    def test_image_uses_vision_fallback(self):
        settings = SimpleNamespace(ocr_provider="vision")
        with patch("document_parser.get_settings", return_value=settings), patch(
            "document_parser.recognize_image", AsyncMock(return_value="白细胞 5.2")
        ) as recognize:
            result = asyncio.run(parse_document(b"image", "image/png", "report"))

        self.assertEqual(result.provider, "vision")
        self.assertEqual(result.text, "白细胞 5.2")
        recognize.assert_awaited_once()

    def test_pdf_requires_mineru_configuration(self):
        settings = SimpleNamespace(ocr_provider="vision")
        with patch("document_parser.get_settings", return_value=settings), self.assertRaises(DocumentParseError):
            asyncio.run(parse_document(b"%PDF-1.7", "application/pdf", "report"))


if __name__ == "__main__":
    unittest.main()
