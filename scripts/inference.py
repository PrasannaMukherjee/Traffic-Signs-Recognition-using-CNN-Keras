import os
import numpy as np
import tensorflow as tf
from PIL import Image
import argparse

def run(args):
    model_dir = args.model_dir
    image_path = args.image_path
    model_path = os.path.join(model_dir, 'best_model.h5')
    
    print("----------- Starting Inference -----------")
    
    if not image_path:
        print("Error: No image path provided. Use --image_path <path>")
        return
    
    # Load trained model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # Load + preprocess image
    print(f"Loading and processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image = image.resize((32, 32))  # FIXED SIZE
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    print("Predicting...")
    prediction = model.predict(image_array)
    predicted_class = int(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction))

    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")

    print("----------- Inference Complete -----------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Traffic Sign Recognition")
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--image_path', type=str, required=True)
    args = parser.parse_args()
    run(args)

