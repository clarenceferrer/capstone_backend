from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
import io
import json

# ----------------------------
# Config: Class labels
# ----------------------------
CLASS_NAMES = ["chair", "door", "person", "stairs", "table"]

# ----------------------------
# Load TFLite model
# ----------------------------
print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path="best_float32.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded. Input details:", input_details)
print("Output details:", output_details)

# ----------------------------
# FastAPI app setup
# ----------------------------
app = FastAPI(title="YOLOv8 TFLite API for Navigation App")

# Allow React Native app to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Later restrict to your frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Image preprocessing
# ----------------------------
def preprocess_image(img):
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb / 255.0
    input_tensor = np.expand_dims(img_norm, axis=0).astype(np.float32)
    return input_tensor

# ----------------------------
# Run inference
# ----------------------------
def run_inference(img):
    input_tensor = preprocess_image(img)
    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    return output

# ----------------------------
# Decode YOLOv8 TFLite output
# ----------------------------
def decode_output(output, conf_thres=0.25, iou_thres=0.45):
    output = np.squeeze(output)
    if output.shape[0] < output.shape[1]:
        output = output.T

    boxes, scores, class_ids = [], [], []

    for row in output:
        x, y, w, h = row[:4]
        class_probs = row[4:]
        class_id = int(np.argmax(class_probs))
        score = float(class_probs[class_id])
        if score < conf_thres:
            continue
        x1 = float(x - w / 2)
        y1 = float(y - h / 2)
        boxes.append([x1, y1, float(w), float(h)])
        scores.append(score)
        class_ids.append(class_id)

    detections = []

    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, score_threshold=conf_thres, nms_threshold=iou_thres
        )
        if len(indices) > 0:
            for i in indices.flatten():
                label = CLASS_NAMES[class_ids[i]] if class_ids[i] < len(CLASS_NAMES) else "unknown"
                detections.append({
                    "class": class_ids[i],
                    "label": label,
                    "confidence": float(scores[i]),
                    "bbox": boxes[i],
                })
    return detections

# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def home():
    return {"message": "🚀 YOLOv8 TFLite API is running! Send POST /detect with an image."}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image.")

    output = run_inference(img)
    detections = decode_output(output)

    # 🔥 Optional: log to console
    print("\n===== DETECTIONS =====")
    print(json.dumps(detections, indent=2))
    print("======================\n")

    return {"detections": detections}
