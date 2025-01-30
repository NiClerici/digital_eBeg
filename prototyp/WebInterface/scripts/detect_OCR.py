from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import cv2
import os
import numpy as np
from detectron2.data import MetadataCatalog
from paddleocr import PaddleOCR

# Function to preprocess the image
def preprocess_image(image, scale_factor=1.5, sharpen=True):
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    if sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        upscaled = cv2.filter2D(upscaled, -1, kernel)

    return upscaled

# Load configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "/Users/nicoclerici/Documents/Bewerbung/DBew/Prototyp_eBew/src/outputs_best/model_final.pth"  # Update with your model path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16
cfg.MODEL.DEVICE = "cpu"
cfg.INPUT.MIN_SIZE_TEST = 1024
cfg.INPUT.MAX_SIZE_TEST = 2048
cfg.TEST.DETECTIONS_PER_IMAGE = 100
cfg.MODEL.RPN.NMS_THRESH = 0.7
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

# Define dataset name and class names
dataset_name = "train_dataset"
MetadataCatalog.get(dataset_name).set(thing_classes=[
    "Strassenname", "Gebaeude Neu", "Gebaeude Bestehend",
    "GeometerStempel", "Nordpfeil", "Massstab", "Parzellengrenze", "Masslinie",
    "Unterschriften", "Titelinformation", "Gebaeude Untergrund", "Legende", "M.ü.M", "Gebaeude Abbruch", "Parzellennummer", "Gebaeudenummer"
])

# Create predictor
predictor = DefaultPredictor(cfg)

# Initialize PaddleOCR
ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

def process_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Error: Image at '{file_path}' could not be loaded.")

    image = preprocess_image(image, scale_factor=3)
    outputs = predictor(image)
    metadata = MetadataCatalog.get(dataset_name)
    instances = outputs["instances"].to("cpu")

    relevant_classes = ["Masslinie", "Parzellennummer", "Gebaeudenummer"]
    extracted_texts = []

    if len(instances) > 0:
        pred_boxes = instances.pred_boxes.tensor.numpy()
        pred_classes = instances.pred_classes.numpy()
        scores = instances.scores.numpy()

        for i, (box, cls, score) in enumerate(zip(pred_boxes, pred_classes, scores)):
            class_name = metadata.thing_classes[cls]
            if class_name in relevant_classes:
                x1, y1, x2, y2 = map(int, box)
                roi = image[y1:y2, x1:x2]
                temp_path = f"/tmp/roi_temp_{i}.png"
                cv2.imwrite(temp_path, roi)

                results = ocr_model.ocr(temp_path, cls=True)
                if results and len(results[0]) > 0:
                    for line in results[0]:
                        text = line[1][0]
                        confidence = line[1][1]
                        extracted_texts.append(f"{class_name}: {text} (Probability: {confidence:.2f})")
                os.remove(temp_path)

    return extracted_texts