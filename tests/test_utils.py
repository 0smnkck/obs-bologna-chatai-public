import logging
from src.utils import normalize_text, setup_logger

def test_normalize_text():
    assert normalize_text("Dr. Öğr. Üyesi Muhammet Talha KAKIZ") == "dr. ogr. uyesi muhammet talha kakiz"
    assert normalize_text("Kakız") == "kakiz"
    assert normalize_text("İıŞşĞğÇçÖöÜü") == "iissggccoouu"
    assert normalize_text("") == ""
    assert normalize_text(None) == ""

def test_setup_logger():
    logger = setup_logger("TestLogger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "TestLogger"
