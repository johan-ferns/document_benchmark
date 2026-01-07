import pypdfium2 as pdfium
from pathlib import Path
from PIL import Image
import io
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor
from surya.settings import settings as surya_settings
from data_structure import Extracted, ExtractedType, Coordinates, ExtractionResults


def pdf_page_to_image(pdf_path: str | Path, page_num: int) -> Image.Image:
    """Convert a PDF page to a PIL Image."""
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[page_num]
    
    # Render at higher resolution for better OCR
    bitmap = page.render(scale=2.0)
    pil_image = bitmap.to_pil()
    
    pdf.close()
    return pil_image


def extract_pdf_content(pdf_path: str | Path) -> ExtractionResults:
    """
    Extract content from a PDF file using Surya and populate Extracted pydantic objects.
    
    Args:
        pdf_path:  Path to the PDF file
        
    Returns:
        ExtractionResults containing lists of extracted texts, tables, and figures
    """
    pdf_path = Path(pdf_path)
    texts = []
    tables = []
    figures = []
    
    # Initialize Surya predictors
    foundation_predictor = FoundationPredictor()
    recognition_predictor = RecognitionPredictor(foundation_predictor)
    detection_predictor = DetectionPredictor()
    layout_predictor = LayoutPredictor(
        FoundationPredictor(checkpoint=surya_settings.LAYOUT_MODEL_CHECKPOINT)
    )
    
    # Get number of pages
    pdf = pdfium.PdfDocument(pdf_path)
    num_pages = len(pdf)
    pdf.close()
    
    # Process each page
    for page_num in range(num_pages):
        # Convert PDF page to image
        page_image = pdf_page_to_image(pdf_path, page_num)
        
        # Run layout analysis to detect different elements
        layout_predictions = layout_predictor([page_image])
        layout_result = layout_predictions[0]
        
        # Run OCR for text extraction
        ocr_predictions = recognition_predictor(
            [page_image], 
            det_predictor=detection_predictor
        )
        ocr_result = ocr_predictions[0]
        
        # Extract text with bounding boxes
        if hasattr(ocr_result, 'text_lines') and ocr_result.text_lines:
            # Combine all text from the page
            full_text = "\n".join([line.text for line in ocr_result.text_lines])
            
            # Get page dimensions
            page_bbox = ocr_result.image_bbox
            
            text_extracted = Extracted(
                extracted_type=ExtractedType. TEXT,
                page_no=page_num + 1,
                path=pdf_path,
                coordinates=Coordinates(
                    x1=int(page_bbox[0]),
                    y1=int(page_bbox[1]),
                    x2=int(page_bbox[2]),
                    y2=int(page_bbox[3])
                ),
                data=full_text,
                data_type="text"
            )
            texts.append(text_extracted)
        
        # Process layout predictions to extract tables and figures
        if hasattr(layout_result, 'bboxes') and layout_result.bboxes:
            for bbox_info in layout_result.bboxes:
                label = bbox_info.label
                bbox = bbox_info.bbox
                
                # Extract tables
                if label == 'Table':
                    table_extracted = Extracted(
                        extracted_type=ExtractedType. TABLE,
                        page_no=page_num + 1,
                        path=pdf_path,
                        coordinates=Coordinates(
                            x1=int(bbox[0]),
                            y1=int(bbox[1]),
                            x2=int(bbox[2]),
                            y2=int(bbox[3])
                        ),
                        data=None,  # Could extract text from this region if needed
                        data_type="text"
                    )
                    tables.append(table_extracted)
                
                # Extract figures (images, charts, pictures)
                elif label in ['Picture', 'Figure', 'Formula']: 
                    figure_extracted = Extracted(
                        extracted_type=ExtractedType.FIGURE,
                        page_no=page_num + 1,
                        path=pdf_path,
                        coordinates=Coordinates(
                            x1=int(bbox[0]),
                            y1=int(bbox[1]),
                            x2=int(bbox[2]),
                            y2=int(bbox[3])
                        ),
                        data=None,
                        data_type="image/png"
                    )
                    figures.append(figure_extracted)
    
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
    
    # Print figure locations for debugging
    print("\nFigure details:")
    for fig in results.figures:
        print(f"  Page {fig.page_no}: {fig.coordinates}")
    
    print("\nTable details:")
    for table in results.tables:
        print(f"  Page {table.page_no}: {table. coordinates}")