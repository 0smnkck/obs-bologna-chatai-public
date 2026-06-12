from src.metrics import MetricsCollector
from langchain_core.documents import Document

def test_metrics_collector():
    collector = MetricsCollector(model_name="test-model", gpu_name="test-gpu")
    
    # Mock data
    doc1 = Document(page_content="Bilgisayar mühendisliği öğrencileri yazılım ve donanım dersleri alırlar.")
    doc2 = Document(page_content="Osmaniye Korkut Ata Üniversitesi Karacaoğlan Yerleşkesi'ndedir.")
    
    collector.start_inference(query="Bilgisayar mühendisliği nerede?", context_token_count=10, retrieved_doc_count=2)
    collector.record_first_token()
    collector.record_token("Yazılım")
    collector.record_token("dersleri")
    collector.end_inference(full_response="Yazılım dersleri verilmektedir.", context_docs=[doc1, doc2])
    
    # Verify faithfulness
    score = collector.compute_faithfulness()
    assert score > 0.0
    
    # Build record
    record = collector.build_record()
    assert record["model_name"] == "test-model"
    assert record["gpu"] == "test-gpu"
    assert record["faithfulness_score"] == score
