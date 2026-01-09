from paddleocr import LayoutDetection

from pathlib import Path
from dotenv import load_dotenv
import json
import os


os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
load_dotenv()
weight_path = os.getenv("WEIGHT_PATH")
print("The weight path still exists : " + str(os.path.exists(weight_path)))

model = LayoutDetection(model_name="PP-DocLayout_plus-L",
                        model_dir=Path(weight_path,"PP-DocLayout_plus-L"))
output = model.predict("data/pdf/output/example_1_page_3.jpg", batch_size=1, layout_nms=True)
for res in output:
    res.print()
    res.save_to_img(save_path="data/pdf/output/")
    # res.save_to_json(save_path="./output/res.json")