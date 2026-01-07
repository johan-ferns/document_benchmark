import os
from pathlib import Path
from paddleocr import PPStructureV3
import pypdfium2 as pdfium
from PIL import Image
from data_structure import Extracted, ExtractedType, Coordinates, ExtractionResults
from tqdm import tqdm
from dotenv import load_dotenv

os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
load_dotenv()

weight_path = os.getenv("WEIGHT_PATH")


def pdf_page_to_image(pdf_path:  str | Path, page_num: int) -> Image.Image:
    """Convert a PDF page to a PIL Image."""
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[page_num]
    
    # Render at good resolution for OCR
    bitmap = page.render(scale=1.0)
    pil_image = bitmap.to_pil()
    
    pdf.close()
    return pil_image


def extract_pdf_content(pdf_path: str | Path) -> ExtractionResults: 
    """
    Extract content from a PDF file using PaddleOCR and populate Extracted pydantic objects.
    
    Args:
        pdf_path:   Path to the PDF file
        
    Returns:
        ExtractionResults containing lists of extracted texts, tables, and figures
    """
    pdf_path = Path(pdf_path)
    texts = []
    tables = []
    figures = []
    
    # Initialize PaddleOCR PP-StructureV3
    # This includes layout analysis, table recognition, and OCR
    pipeline = PPStructureV3(
        # Layout Detection - detects text, table, figure regions
        layout_detection_model_dir=weight_path + "official_models/PP-DocLayout_plus-L",
        
        # Region Detection
        region_detection_model_dir=weight_path + "official_models/PP-DocBlockLayout",

        # Text Detection (OCR) - detects text bounding boxes
        text_detection_model_dir=weight_path + "official_models/PP-OCRv5_server_det",
        
        # Text Recognition (OCR) - recognizes text content
        text_recognition_model_dir=weight_path + "official_models/PP-OCRv5_server_rec",
        
        # Table Recognition (enabled as you requested)
        table_classification_model_dir=weight_path + "official_models/PP-LCNet_x1_0_table_cls",
        wired_table_structure_recognition_model_dir=weight_path + "official_models/SLANeXt_wired",
        wireless_table_structure_recognition_model_dir=weight_path + "official_models/SLANet_plus",
        wired_table_cells_detection_model_dir=weight_path + "official_models/RT-DETR-L_wired_table_cell_det",
        wireless_table_cells_detection_model_dir=weight_path + "official_models/RT-DETR-L_wireless_table_cell_det",
        
        # Optional models (you have them, so including them)
        # doc_orientation_classify_model_dir=weight_path + "official_models/PP-LCNet_x1_0_doc_ori",
        # doc_unwarping_model_dir=weight_path + "official_models/UVDoc",
        # textline_orientation_model_dir=weight_path + "official_models/PP-LCNet_x1_0_textline_ori",
        # formula_recognition_model_dir=weight_path + "official_models/PP-FormulaNet_plus-L",
        # chart_recognition_model_dir=weight_path + "official_models/PP-Chart2Table",
        
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

    
    # Get number of pages
    pdf = pdfium.PdfDocument(pdf_path)
    num_pages = len(pdf)
    pdf.close()
    
    # Process each page
    for page_num in tqdm(range(num_pages)):
        # Convert PDF page to image
        page_image = pdf_page_to_image(pdf_path, page_num)
        
        # Save temporarily as PaddleOCR works with file paths or numpy arrays
        import numpy as np
        img_array = np.array(page_image)
        
        # Run PP-StructureV3 prediction
        result = pipeline. predict(input=img_array)
        
        # Process results
        for res in result:
            # Get layout parsing results
            if hasattr(res, 'layout_parsing_result'):
                layout_result = res.layout_parsing_result
                
                # Iterate through detected regions
                for region in layout_result: 
                    region_type = region.get('type', '').lower()
                    bbox = region.get('bbox', [0, 0, 0, 0])
                    
                    # Extract coordinates
                    coords = Coordinates(
                        x1=int(bbox[0]),
                        y1=int(bbox[1]),
                        x2=int(bbox[2]),
                        y2=int(bbox[3])
                    )
                    
                    # Handle different region types
                    if region_type == 'text':
                        # Extract text content
                        text_content = region.get('text', '')
                        
                        if text_content and text_content.strip():
                            text_extracted = Extracted(
                                extracted_type=ExtractedType. TEXT,
                                page_no=page_num + 1,
                                path=pdf_path,
                                coordinates=coords,
                                data=text_content,
                                data_type="text"
                            )
                            texts.append(text_extracted)
                    
                    elif region_type == 'table': 
                        # Extract table
                        table_html = region.get('html', '')
                        
                        table_extracted = Extracted(
                            extracted_type=ExtractedType.TABLE,
                            page_no=page_num + 1,
                            path=pdf_path,
                            coordinates=coords,
                            data=table_html if table_html else None,
                            data_type="text"
                        )
                        tables.append(table_extracted)
                    
                    elif region_type in ['figure', 'image', 'chart']:
                        # Extract figure/chart
                        figure_extracted = Extracted(
                            extracted_type=ExtractedType. FIGURE,
                            page_no=page_num + 1,
                            path=pdf_path,
                            coordinates=coords,
                            data=None,
                            data_type="image/png"
                        )
                        figures.append(figure_extracted)
            
            # Alternative: if layout_parsing_result is not available, use regions directly
            elif hasattr(res, 'regions'):
                for region in res.regions:
                    bbox = region.get('bbox', [0, 0, 0, 0])
                    region_type = region.get('type', '').lower()
                    
                    coords = Coordinates(
                        x1=int(bbox[0]),
                        y1=int(bbox[1]),
                        x2=int(bbox[2]),
                        y2=int(bbox[3])
                    )
                    
                    if region_type == 'text': 
                        text_content = region. get('res', {}).get('text', '')
                        if text_content: 
                            texts.append(Extracted(
                                extracted_type=ExtractedType.TEXT,
                                page_no=page_num + 1,
                                path=pdf_path,
                                coordinates=coords,
                                data=text_content,
                                data_type="text"
                            ))
                    
                    elif region_type == 'table': 
                        table_html = region. get('res', {}).get('html', '')
                        tables.append(Extracted(
                            extracted_type=ExtractedType.TABLE,
                            page_no=page_num + 1,
                            path=pdf_path,
                            coordinates=coords,
                            data=table_html if table_html else None,
                            data_type="text"
                        ))
                    
                    elif region_type in ['figure', 'image']: 
                        figures.append(Extracted(
                            extracted_type=ExtractedType.FIGURE,
                            page_no=page_num + 1,
                            path=pdf_path,
                            coordinates=coords,
                            data=None,
                            data_type="image/png"
                        ))
    
    return ExtractionResults(
        texts=texts,
        tables=tables,
        figures=figures
    )


if __name__ == "__main__": 
    # Example usage
    pdf_file = "data/pdf/input/example_1.pdf"
    results = extract_pdf_content(pdf_file)
    
    print(f"Extracted {len(results.texts)} text blocks")
    print(f"Extracted {len(results.tables)} tables")
    print(f"Extracted {len(results.figures)} figures")
    
    # Print details
    print("\nText blocks:")
    for text in results.texts:
        print(f"  Page {text.page_no}: {text.coordinates}")
    
    print("\nTables:")
    for table in results.tables:
        print(f"  Page {table.page_no}: {table.coordinates}")
    
    print("\nFigures:")
    for fig in results.figures:
        print(f"  Page {fig. page_no}: {fig.coordinates}")

