from data_structure import Extracted, ExtractedType, Coordinates, ExtractionResults
from pathlib import Path
import json


def calculate_iou(box1 : Coordinates, box2 : Coordinates):
    """
    Calculate Intersection over Union for two bounding boxes
    
    Args:
        box1, box2: Tuples of (x1, y1, x2, y2)
    
    Returns:
        IOU value between 0 and 1
    """
    # Calculate intersection coordinates
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)
    
    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union area
    area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
    area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def calculate_containment(box1, box2):
    """
    Calculate what percentage of box1 is contained within box2.
    This is more flexible than IOU for cases where one box is slightly 
    larger than the other. 
    
    Args:
        box1: Tuple of (x1, y1, x2, y2) - the box we're checking containment for
        box2: Tuple of (x1, y1, x2, y2) - the potential container box
    
    Returns:
        Containment ratio between 0 and 1 (1. 0 = box1 is fully inside box2)
    """
    # Calculate intersection coordinates
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)
    
    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate area of box1
    area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)

    # Return what fraction of box1 is contained in the intersection
    return intersection / area1 if area1 > 0 else 0



if __name__ == "__main__":  

    json_path = Path("data/pdf/outputs/example_1_comp.json")

    with open(json_path, 'r', encoding='utf-8') as f:
        results_dict = json.load(f)

    # Convert path strings back to Path objects
    for category in ['texts', 'tables', 'figures']:
        for item in results_dict[category]: 
            if item['path'] is not None:
                item['path'] = Path(item['path'])

    # Create ExtractionResults from dictionary
    gt_results = ExtractionResults(**results_dict)

    print(len(gt_results.texts))