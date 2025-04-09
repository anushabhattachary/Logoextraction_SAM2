# scripts/segment_logos_sam.py

import os
import cv2
import numpy as np

from segment_anything import sam_model_registry, SamPredictor

def load_sam():
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
    sam.to("cuda")
    return SamPredictor(sam)

def segment_and_save(predictor, image_path, output_path):
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)
    masks, _, _ = predictor.predict(box=None, multimask_output=True)
    
    if masks:
        mask = masks[0].astype(np.uint8)
        logo = cv2.bitwise_and(rgb, rgb, mask=mask)
        cv2.imwrite(output_path, cv2.cvtColor(logo, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved logo to {output_path}")
