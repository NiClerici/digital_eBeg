from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import os
from detect_OCR import preprocess_image, predictor, ocr_model, MetadataCatalog

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        file_path = os.path.join('/tmp', file.filename)
        file.save(file_path)

        # Process the image
        image = cv2.imread(file_path)
        image = preprocess_image(image, scale_factor=3)
        outputs = predictor(image)
        metadata = MetadataCatalog.get("train_dataset")
        instances = outputs["instances"].to("cpu")

        # Extract relevant data
        relevant_classes = ["Masslinie", "Parzellennummer", "Gebaeudenummer"]
        extracted_texts = []
        if len(instances) > 0:
            pred_boxes = instances.pred_boxes.tensor.numpy()
            pred_classes = instances.pred_classes.numpy()
            scores = instances.scores.numpy()
            total_instances = len(pred_boxes)
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
                    # Send progress update
                    progress = int((i + 1) / total_instances * 100)
                    socketio.emit('progress', {'progress': progress})

        return jsonify({'extracted_texts': extracted_texts})

@app.route('/overview', methods=['POST'])
def overview():
    form_data = request.json
    # Combine form data with processed image data
    combined_data = form_data.copy()
    combined_data.update(request.json)
    return jsonify(combined_data)

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, port=5000)