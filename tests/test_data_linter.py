import os
import json
import tempfile
import shutil
from src.data_linter import lint_data

def test_lint_data():
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a valid JSON file
        valid_data = {
            "title": "BMB101 - Intro to Programming",
            "content": "A detailed content that is more than five hundred characters long to pass the length check. " * 10,
            "metadata": {
                "ders_kodu": "BMB101",
                "ders_adi": "Intro to Programming",
                "ogretim_uyesi": "Dr. John Doe"
            }
        }
        with open(os.path.join(temp_dir, "bmb101_bologna.json"), "w", encoding="utf-8") as f:
            json.dump(valid_data, f)

        # Create an invalid JSON file (missing fields)
        invalid_data = {
            "title": "BMB102",
            "content": "Short text"
        }
        with open(os.path.join(temp_dir, "bmb102_bologna.json"), "w", encoding="utf-8") as f:
            json.dump(invalid_data, f)

        # Run linter
        res = lint_data(temp_dir)
        
        assert res["total_files"] == 2
        assert res["valid_files"] == 1
        assert len(res["errors"]) == 1  # bmb102 has okuma/missing fields errors
        assert "bmb102_bologna.json" in res["errors"][0]
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
