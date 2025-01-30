import cv2
import requests
import matplotlib.pyplot as plt
import numpy as np
import os
from flask import Flask, request, send_file, jsonify, render_template
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from io import BytesIO

# Flask setup
app = Flask(__name__,
            template_folder="/Users/nicoclerici/Documents/Bewerbung/DBew/Prototyp_eBew"
                            "/prototyp/WebInterface/templates", static_folder='static')

# Upload folder setup
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static/uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Detectron2 configuration setup
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "/Users/nicoclerici/Documents/Bewerbung/DBew/Prototyp_eBew/prototyp/WebInterface/cnn/model_final.pth"  # Change this to the correct path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16
cfg.MODEL.DEVICE = "cpu"  # Use "cuda" for GPU
cfg.INPUT.MIN_SIZE_TEST = 1024
cfg.INPUT.MAX_SIZE_TEST = 2048
cfg.TEST.DETECTIONS_PER_IMAGE = 50
cfg.MODEL.RPN.NMS_THRESH = 0.7
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

# Metadata setup
dataset_name = "train_dataset"
MetadataCatalog.get(dataset_name).set(thing_classes=[
    "Strassenname", "Gebaeude Neu", "Gebaeude Bestehend",
    "GeometerStempel", "Nordpfeil", "Massstab", "Parzellengrenze", "Masslinie",
    "Unterschriften", "Titelinformation", "Gebaeude Untergrund", "Legende", "M.ü.M", "Gebaeude Abbruch",
    "Parzellennummer", "Gebaeudenummer"
])
metadata = MetadataCatalog.get(dataset_name)
predictor = DefaultPredictor(cfg)


@app.route('/upload', methods=['POST'])
def upload_file():
    # Fetch the file from the request
    file = request.files.get('file')
    if not file:
        return "No file part in the request.", 400

    # Check if the file name is empty
    if not file.filename:
        return "No selected file", 400

    # Validate file extension and MIME type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return "Unsupported file format. Please upload PNG or JPEG images only.", 400
    if file.mimetype not in ['image/jpeg', 'image/png']:
        return "Unsupported file type. Please upload a valid PNG or JPEG image.", 400

    # Save the original file in the upload folder
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Use the saved file path to load the image with OpenCV
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if image is None:
        return "Unable to process the image. Please upload a valid PNG or JPEG file.", 400

    # Model inference and image processing
    outputs = predictor(image)

    if outputs["instances"].has("pred_boxes") and len(outputs["instances"]) > 0:
        # Visualize predictions
        v = Visualizer(image[:, :, ::-1], metadata, scale=2.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Save processed image
        processed_file_path = os.path.join(
            app.config['UPLOAD_FOLDER'], "processed_" + file.filename
        )
        cv2.imwrite(processed_file_path, out.get_image()[:, :, ::-1])

        # Send JSON response
        return jsonify({
            "original": f"/static/uploads/{file.filename}",
            "processed": f"/static/uploads/processed_{file.filename}"
        })

    return "No objects detected in the uploaded image.", 200


@app.route('/gallery')
def image_gallery():
    # Load all files from the upload folder
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    files = [f"/static/uploads/{file}" for file in files]  # Dynamic paths for templates

    return render_template('image_gallery.html', files=files)



@app.route('/')
def serve_html():
    return render_template('frist_try.html')


# Run the Flask server
if __name__ == "__main__":
    app.run(debug=True)
