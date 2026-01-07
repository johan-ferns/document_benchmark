
import os
from pathlib import Path
from paddleocr import PPStructureV3

pipeline = PPStructureV3(
    # Layout Detection - detects text, table, figure regions
    layout_detection_model_dir="C:/Users/fernjoh/.paddlex/official_models/PP-DocLayout_plus-L",
    
    # Region Detection
    region_detection_model_dir="C:/Users/fernjoh/.paddlex/official_models/PP-DocBlockLayout",

    # Text Detection (OCR) - detects text bounding boxes
    text_detection_model_dir="C:/Users/fernjoh/.paddlex/official_models/PP-OCRv5_server_det",
    
    # Text Recognition (OCR) - recognizes text content
    text_recognition_model_dir="C:/Users/fernjoh/.paddlex/official_models/PP-OCRv5_server_rec",
    
    # Table Recognition (enabled as you requested)
    table_classification_model_dir="C:/Users/fernjoh/.paddlex/official_models/PP-LCNet_x1_0_table_cls",
    wired_table_structure_recognition_model_dir="C:/Users/fernjoh/.paddlex/official_models/SLANeXt_wired",
    wireless_table_structure_recognition_model_dir="C:/Users/fernjoh/.paddlex/official_models/SLANet_plus",
    wired_table_cells_detection_model_dir="C:/Users/fernjoh/.paddlex/official_models/RT-DETR-L_wired_table_cell_det",
    wireless_table_cells_detection_model_dir="C:/Users/fernjoh/.paddlex/official_models/RT-DETR-L_wireless_table_cell_det",
    
    # Optional models (you have them, so including them)
    # doc_orientation_classify_model_dir="C:/Users/fernjoh/.paddlex/official_models/PP-LCNet_x1_0_doc_ori",
    # doc_unwarping_model_dir="C:/Users/fernjoh/.paddlex/official_models/UVDoc",
    # textline_orientation_model_dir="C:/Users/fernjoh/.paddlex/official_models/PP-LCNet_x1_0_textline_ori",
    # formula_recognition_model_dir="C:/Users/fernjoh/.paddlex/official_models/PP-FormulaNet_plus-L",
    # chart_recognition_model_dir="C:/Users/fernjoh/.paddlex/official_models/PP-Chart2Table",
    
    # Enable table recognition
    use_table_recognition=True,
    use_region_detection=True,  # Set True if you have PP-DocBlockLayout

    # Disable features you don't need (optional - enable if you want)
    use_doc_orientation_classify=False,  # Set True if you want document rotation correction
    use_doc_unwarping=False,  # Set True if you want document unwarping
    use_textline_orientation=False,  # Set True if you want text line orientation
    use_seal_recognition=False,  # No seal models available
    use_formula_recognition=False,  # Set True if you need formula recognition
    use_chart_recognition=False,  # Set True if you need chart to table conversion

)