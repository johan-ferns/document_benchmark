import os
from pathlib import Path
from paddleocr import PPStructureV3
import pypdfium2 as pdfium
from PIL import Image
from data_structure import Extracted, ExtractedType, Coordinates, ExtractionResults
from tqdm import tqdm
from dotenv import load_dotenv
import json
from copy import deepcopy



os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
load_dotenv()

weight_path = os.getenv("WEIGHT_PATH")


def pdf_page_to_image(pdf_path:   str | Path, page_num:   int) -> Image.Image:
    """Convert a PDF page to a PIL Image."""
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[page_num]
    
    # Render at good resolution for OCR
    bitmap = page.render(scale=2.0)
    pil_image = bitmap.to_pil()
    
    pdf.close()
    return pil_image


def extract_pdf_content(pdf_path:  str | Path, debug:  bool = False) -> ExtractionResults:   
    """
    Extract content from a PDF file using PaddleOCR and populate Extracted pydantic objects.
    
    Args:
        pdf_path:    Path to the PDF file
        debug:  If True, print detailed debugging information
        
    Returns:  
        ExtractionResults containing lists of extracted texts, tables, and figures
    """
    pdf_path = Path(pdf_path)
    texts = []
    tables = []
    figures = []
    
    # Initialize PaddleOCR PP-StructureV3
    pipeline = PPStructureV3(
        # Layout Detection - detects text, table, figure regions
        layout_detection_model_dir=weight_path + "official_models/PP-DocLayout_plus-L",
        
        # Region Detection
        region_detection_model_dir=weight_path + "official_models/PP-DocBlockLayout",

        # Text Detection (OCR) - detects text bounding boxes
        text_detection_model_dir=weight_path + "official_models/PP-OCRv5_server_det",
        
        # Text Recognition (OCR) - recognizes text content
        text_recognition_model_dir=weight_path + "official_models/PP-OCRv5_server_rec",
        
        # Table Recognition
        table_classification_model_dir=weight_path + "official_models/PP-LCNet_x1_0_table_cls",
        wired_table_structure_recognition_model_dir=weight_path + "official_models/SLANeXt_wired",
        wireless_table_structure_recognition_model_dir=weight_path + "official_models/SLANet_plus",
        wired_table_cells_detection_model_dir=weight_path + "official_models/RT-DETR-L_wired_table_cell_det",
        wireless_table_cells_detection_model_dir=weight_path + "official_models/RT-DETR-L_wireless_table_cell_det",
        
        # Enable table recognition
        use_table_recognition=True,
        use_region_detection=True,

        # Disable features you don't need
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_seal_recognition=False,
        use_formula_recognition=False,
        use_chart_recognition=False,
    )

    
    # Get number of pages
    pdf = pdfium.PdfDocument(pdf_path)
    num_pages = len(pdf)
    pdf.close()
    
    print(f"Processing {num_pages} pages...")
    
    # Process each page
    for page_num in tqdm(range(num_pages)):
        # Convert PDF page to image
        page_image = pdf_page_to_image(pdf_path, page_num)
        
        # Convert to numpy array
        import numpy as np
        img_array = np.array(page_image)
        
        if debug:
            print(f"\n=== Page {page_num + 1} ===")
            print(f"Image shape:   {img_array.shape}")
        
        # Run PP-StructureV3 prediction
        result = pipeline.predict(input=img_array)
        
        # Process results
        for res in result:
            # Get the parsing_res_list which contains the layout regions
            if 'parsing_res_list' in res:
                parsing_results = res['parsing_res_list']
                
                if debug:  
                    print(f"Found {len(parsing_results)} regions in parsing_res_list")
                
                for region in parsing_results:
                    # Access LayoutBlock attributes using the correct attribute names
                    region_label = getattr(region, 'label', 'unknown')
                    bbox = getattr(region, 'bbox', None)
                    content = getattr(region, 'content', '')
                    
                    if debug:
                        print(f"  Region label: {region_label}, bbox: {bbox}")
                    
                    if bbox is None or len(bbox) < 4:
                        if debug:
                            print(f"  Skipping region - invalid bbox")
                        continue
                    
                    # Extract coordinates
                    coords = Coordinates(
                        x1=int(bbox[0]),
                        y1=int(bbox[1]),
                        x2=int(bbox[2]),
                        y2=int(bbox[3])
                    )
                    
                    # Categorize based on label
                    # TEXT types
                    # # Dropping classes : 
                    # ['doc_title',  'section_title', 'header', 'footnote', 'footer', 'number',
                    #  'vision_footnote']
                    if region_label in ['paragraph_title',
                                       'text', 'list_item', 'equation',
                                       ]: 
                        if content and content.strip():
                            if debug:
                                print(f"    → TEXT: {content[: 50]}...")
                            
                            text_extracted = Extracted(
                                extracted_type=ExtractedType. TEXT,
                                page_no=page_num + 1,
                                path=pdf_path,
                                coordinates=coords,
                                data=content,
                                data_type="text"
                            )
                            texts.append(text_extracted)
                    
                    # TABLE types
                    elif region_label in ['table', 'table_caption', 'table_footnote']: 
                        # Try to get HTML if available
                        table_html = getattr(region, 'html', '')
                        
                        if debug:
                            print(f"    → TABLE (has HTML:  {bool(table_html)})")
                        
                        table_extracted = Extracted(
                            extracted_type=ExtractedType.TABLE,
                            page_no=page_num + 1,
                            path=pdf_path,
                            coordinates=coords,
                            data=table_html if table_html else content,
                            data_type="text"
                        )
                        tables.append(table_extracted)
                    
                    # FIGURE types
                    #:TODO  may need to be added.
                    # # Dropping classes : 
                    # ['figure_title',  'image']
                    elif region_label in ['figure', 'chart', 'figure_caption']:
                        if debug:
                            print(f"    → FIGURE/CHART")
                        
                        figure_extracted = Extracted(
                            extracted_type=ExtractedType.FIGURE,
                            page_no=page_num + 1,
                            path=pdf_path,
                            coordinates=coords,
                            data=None,  # Could store the image data if needed
                            data_type="image/png"
                        )
                        figures.append(figure_extracted)
                    
                    else:
                        if debug: 
                            print(f"    → UNKNOWN LABEL: {region_label}")
                            if content: 
                                print(f"       Content: {content[:50]}...")
    
    return ExtractionResults(
        texts=texts,
        tables=tables,
        figures=figures
    )



def rescale_extraction_results(
    results: ExtractionResults,
    original_width: int,
    original_height: int,
    target_width: int = 612,
    target_height: int = 792
) -> ExtractionResults:
    """
    Rescale all bounding boxes in ExtractionResults to fit a target size.
    
    Args:
        results: Original ExtractionResults
        original_width: Width of the original image/page
        original_height: Height of the original image/page
        target_width: Target width (default: 612 for PDF points)
        target_height: Target height (default: 792 for PDF points)
        
    Returns:
        New ExtractionResults with rescaled coordinates
    """
    # Calculate scaling factors
    width_scale = target_width / original_width
    height_scale = target_height / original_height
    
    print(f"Rescaling from ({original_width}x{original_height}) to ({target_width}x{target_height})")
    print(f"Scale factors: width={width_scale:.4f}, height={height_scale:.4f}")
    
    # Create a deep copy to avoid modifying the original
    rescaled_results = deepcopy(results)
    
    # Rescale all categories
    for category in [rescaled_results.texts, rescaled_results.tables, rescaled_results.figures]:
        for item in category:
            if item. coordinates is not None:
                # Rescale coordinates
                item.coordinates = Coordinates(
                    x1=int(item.coordinates.x1 * width_scale),
                    y1=int(item.coordinates.y1 * height_scale),
                    x2=int(item.coordinates.x2 * width_scale),
                    y2=int(item.coordinates.y2 * height_scale)
                )
    
    return rescaled_results


if __name__ == "__main__":  
    # Example usage
    example_file_name = "example_2"
    pdf_file = "data/pdf/input/"+ example_file_name +".pdf"
    
    if os.path.exists("data/pdf/output/"+ example_file_name +"/") == False:
        os.mkdir("data/pdf/output/"+ example_file_name +"/")
    output_path_org_result = "data/pdf/output/"+ example_file_name +"/"+ example_file_name +"_org.json"
    output_path_comp_result = "data/pdf/output/"+ example_file_name +"/"+ example_file_name +"_gt.json"

    # Get content
    results = extract_pdf_content(pdf_file, debug=True)
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS:")
    print(f"{'='*50}")
    print(f"Extracted {len(results.texts)} text blocks")
    print(f"Extracted {len(results.tables)} tables")
    print(f"Extracted {len(results.figures)} figures")
    
    # Print details
    if results.texts:
        print("\n📝 Text blocks:")
        for i, text in enumerate(results.texts[:5]):  # Show first 5
            print(f"  [{i+1}] Page {text.page_no}: {text.coordinates}")
            print(f"      Preview: {str(text.data)[:100]}...")
    else:
        print("\n📝 No text blocks found!")
    
    if results.tables:
        print("\n📊 Tables:")
        for i, table in enumerate(results.tables):
            print(f"  [{i+1}] Page {table.page_no}: {table. coordinates}")
    else:
        print("\n📊 No tables found!")
    
    if results.figures:
        print("\n📈 Figures:")
        for i, fig in enumerate(results.figures):
            print(f"  [{i+1}] Page {fig.page_no}: {fig. coordinates}")
    else:
        print("\n📈 No figures found!")

    # Convert to dictionary using Pydantic's model_dump
    results_dict = results.model_dump(mode='json')

    # Save to original image size to JSON file
    with open(output_path_org_result, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Saved extraction results to {output_path_org_result}")
    


    # Get the actual PDF page dimensions
    pdf = pdfium.PdfDocument(pdf_file)
    page = pdf[0]  # Get first page for dimensions
    pdf_width = page.get_width()
    pdf_height = page.get_height()
    pdf.close()

    # # Render at good resolution for OCR
    # bitmap = page.render(scale=2.0)
    # pil_image = bitmap.to_pil()

    # Calculate rendered image dimensions (at scale=2.0)
    render_scale = 2.0
    rendered_width = int(pdf_width * render_scale)
    rendered_height = int(pdf_height * render_scale)

    # Rescale back to PDF dimensions
    rescaled_results = rescale_extraction_results(
        results,
        original_width=rendered_width,
        original_height=rendered_height,
        target_width=int(pdf_width),
        target_height=int(pdf_height)
    )

    # Convert to dictionary using Pydantic's model_dump
    rescaled_results_dict = rescaled_results.model_dump(mode='json')

    # Save to rescaled image size to JSON file
    with open(output_path_comp_result, 'w', encoding='utf-8') as f:
        json.dump(rescaled_results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Saved rescaled extraction results to {output_path_comp_result}")