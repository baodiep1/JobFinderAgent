import os
import unittest
from tools.pdf_extractor import pdf_extractor_tool

class TestPDFExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_pdf = os.path.join("test_files", "sample_cv.pdf")
        if not os.path.exists(cls.test_pdf):
            raise FileNotFoundError(f"Test PDF missing: {cls.test_pdf}")

    def test_pdf_extraction(self):
        text = pdf_extractor_tool(self.test_pdf)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 10)
        print(f"\nPDF Extraction Test Passed. Extracted {len(text)} characters.")

if __name__ == "__main__":
    unittest.main()
