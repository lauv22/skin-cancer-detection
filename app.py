import os
import uuid
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
import cv2
import base64

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "best_model.keras"
UPLOAD_FOLDER = "static/uploads"
IMG_SIZE     = (224, 224)
CLASS_NAMES  = ["Benign", "Malignant"]
LAST_CONV    = "conv5_block3_out"

# ── Load model once at startup ────────────────────────────────────────────────
print("Loading model...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded ✅")

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def compute_gradcam(model, img_array):
    resnet_sub   = model.get_layer("resnet50")
    last_conv_out = resnet_sub.get_layer(LAST_CONV).output
    grad_model   = keras.Model(
        inputs=resnet_sub.input,
        outputs=[last_conv_out, resnet_sub.output],
    )
    with tf.GradientTape() as tape:
        img_tensor = tf.cast(img_array, tf.float32)
        conv_outputs, resnet_out = grad_model(img_tensor, training=False)
        gap    = model.get_layer("gap")(resnet_out)
        d512   = model.get_layer("dense_512")(gap)
        d128   = model.get_layer("dense_128")(d512)
        pred   = model.get_layer("output")(d128)
        loss   = pred[:, 0]
    grads        = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0.0)
    heatmap      = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def apply_overlay(original_img, heatmap, alpha=0.45):
    h, w            = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb     = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(
        original_img.astype(np.uint8), 1.0 - alpha,
        heatmap_rgb, alpha, 0,
    )
    return overlay


def array_to_base64(img_array):
    img     = Image.fromarray(img_array.astype(np.uint8))
    import io
    buffer  = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def predict(image_path):
    pil_img   = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    img_orig  = np.array(pil_img)
    img_pre   = resnet_preprocess(img_orig.astype(np.float32))
    img_input = np.expand_dims(img_pre, axis=0)

    prob      = float(model.predict(img_input, verbose=0)[0][0])
    label     = int(prob >= 0.5)
    pred_class = CLASS_NAMES[label]

    if label == 0:
        confidence = (1 - prob) * 100
    else:
        confidence = prob * 100

    heatmap = compute_gradcam(model, img_input)
    overlay = apply_overlay(img_orig, heatmap)

    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * cv2.resize(heatmap, IMG_SIZE)), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    return {
        "predicted_class" : pred_class,
        "confidence"      : round(confidence, 2),
        "label"           : label,
        "original_img"    : array_to_base64(img_orig),
        "heatmap_img"     : array_to_base64(heatmap_rgb),
        "overlay_img"     : array_to_base64(overlay),
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save uploaded file with unique name
    ext      = Path(file.filename).suffix
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    result = predict(filepath)

    # Clean up uploaded file
    os.remove(filepath)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False)