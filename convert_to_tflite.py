import tensorflow as tf

# Load the trained Keras model
model_path = './model_trained.keras'
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Export the model as a SavedModel
saved_model_path = './saved_model'
try:
    model.export(saved_model_path)  # Use export for SavedModel
    print("Model exported as SavedModel.")
except Exception as e:
    print(f"Error during exporting model as SavedModel: {e}")
    exit()

# Convert the SavedModel to TFLite
try:
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    tflite_model = converter.convert()
    print("Model converted to TFLite format successfully.")

    # Save the converted model to a .tflite file
    tflite_model_path = './model.tflite'
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model saved successfully to {tflite_model_path}")
except Exception as e:
    print(f"Error during conversion to TFLite: {e}")
    exit()
