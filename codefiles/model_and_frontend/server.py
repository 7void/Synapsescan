import os
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf

#custom layers for our differential attention model 
class DifferentialAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads=None, key_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim

    def build(self, input_shape):
        self.query_dense = tf.keras.layers.Dense(self.key_dim)
        self.key_dense = tf.keras.layers.Dense(self.key_dim)
        self.value_dense = tf.keras.layers.Dense(self.key_dim)
        super().build(input_shape)

    def call(self, x):
        Q = self.query_dense(x)
        K = self.key_dense(x)
        V = self.value_dense(x)
        scores = tf.einsum('bij,bkj->bik', Q, K)
        weights = tf.nn.softmax(scores, axis=-1)
        mask = tf.cast(weights > tf.reduce_mean(weights, axis=-1, keepdims=True), tf.float32)
        sparse = weights * mask
        return tf.einsum('bij,bjk->bik', sparse, V)

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads, "key_dim": self.key_dim})
        return config

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "inception_attention_finetuned_3.h5")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please place the .h5 file next to server.py.")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"DifferentialAttention": DifferentialAttention}
)

def preprocess_image(img: Image.Image, target_size=(224, 224)) -> np.ndarray:
    img = img.resize(target_size)
    arr = np.array(img) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    return np.expand_dims(arr, 0)

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'no file part'}), 400

    file = request.files['file']
    try:
        img = Image.open(BytesIO(file.read())).convert('RGB')
        x = preprocess_image(img)
        preds = model.predict(x)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])
        return jsonify({'classIndex': idx, 'confidence': round(confidence, 4)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
