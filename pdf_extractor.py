import pypdfium2 as pdfium
from pathlib import Path
from PIL import Image
import io
import json
from data_structure import Extracted, ExtractedType, Coordinates, ExtractionResults


def extract_pdf_content(pdf_path: str | Path) -> ExtractionResults: 
    """
    Extract content from a PDF file and populate Extracted pydantic objects. 
    
    Args:
        pdf_path:  Path to the PDF file
        
    Returns: 
        ExtractionResults containing lists of extracted texts, tables, and figures
    """
    pdf_path = Path(pdf_path)
    texts = []
    tables = []
    figures = []
    
    pdf = pdfium.PdfDocument(pdf_path)
    
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        
        # Extract text
        textpage = page.get_textpage()
        text_content = textpage.get_text_range()
        
        if text_content and text_content. strip():
            width = page. get_width()
            height = page.get_height()
            
            text_extracted = Extracted(
                extracted_type=ExtractedType.TEXT,
                page_no=page_num + 1,
                path=pdf_path,
                coordinates=Coordinates(
                    x1=0,
                    y1=0,
                    x2=int(width),
                    y2=int(height)
                ),
                data=text_content,
                data_type="text"
            )
            texts.append(text_extracted)
        
        # Extract images using pdfplumber for better image detection
        import pdfplumber
        with pdfplumber.open(pdf_path) as plumber_pdf:
            plumber_page = plumber_pdf.pages[page_num]
            
            # Get images from the page
            if hasattr(plumber_page, 'images') and plumber_page.images:
                for img_idx, img in enumerate(plumber_page.images):
                    img_bbox = (
                        int(img.get('x0', 0)),
                        int(img.get('top', 0)),
                        int(img. get('x1', 0)),
                        int(img.get('bottom', 0))
                    )
                    
                    figure_extracted = Extracted(
                        extracted_type=ExtractedType.FIGURE,
                        page_no=page_num + 1,
                        path=pdf_path,
                        coordinates=Coordinates(
                            x1=img_bbox[0],
                            y1=img_bbox[1],
                            x2=img_bbox[2],
                            y2=img_bbox[3]
                        ),
                        data=None,
                        data_type="image/png"
                    )
                    figures.append(figure_extracted)
            
            # Also check for figures (non-text regions that might be charts/diagrams)
            if hasattr(plumber_page, 'figures') and plumber_page.figures:
                for fig_idx, fig in enumerate(plumber_page.figures):
                    fig_bbox = (
                        int(fig.get('x0', 0)),
                        int(fig.get('top', 0)),
                        int(fig.get('x1', 0)),
                        int(fig.get('bottom', 0))
                    )
                    
                    figure_extracted = Extracted(
                        extracted_type=ExtractedType.FIGURE,
                        page_no=page_num + 1,
                        path=pdf_path,
                        coordinates=Coordinates(
                            x1=fig_bbox[0],
                            y1=fig_bbox[1],
                            x2=fig_bbox[2],
                            y2=fig_bbox[3]
                        ),
                        data=None,
                        data_type="image/png"
                    )
                    figures.append(figure_extracted)
    
    pdf.close()
    
    return ExtractionResults(
        texts=texts,
        tables=tables,
        figures=figures
    )


if __name__ == "__main__": 
    # Example usage
    pdf_file = "data/pdf/input/example_1.pdf"
    output_json_path = "data/pdf/output/example_1.json"
    results = extract_pdf_content(pdf_file)
    
    print(f"Extracted {len(results.texts)} text blocks")
    print(f"Extracted {len(results.tables)} tables")
    print(f"Extracted {len(results.figures)} figures")

    # Convert to dictionary using Pydantic's model_dump method
    results_dict = results.model_dump(mode='json')
    
    # Save to JSON file
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    