import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imblearn.over_sampling import RandomOverSampler

#parameters
MODEL_PATH = "inception_attention_model_3.h5"
FINETUNE_PATH = "inception_attention_finetuned_3"
UNFREEZE_COUNT = 60         
FINE_TUNE_LR = 1e-5
FINE_TUNE_EPOCHS = 25
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

dataset_path = "Train_Images"
data = []
for label in os.listdir(dataset_path):
    sub_dir = os.path.join(dataset_path, label)
    if os.path.isdir(sub_dir):
        for fname in os.listdir(sub_dir):
            data.append([os.path.join(sub_dir, fname), label])

df = pd.DataFrame(data, columns=["file_path", "label"])

a = df.copy()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
a['category_encoded'] = le.fit_transform(a['label'])
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(a[['file_path']], a['category_encoded'])
df_res = pd.DataFrame(X_res, columns=['file_path'])
df_res['category_encoded'] = y_res.astype(str)

from sklearn.model_selection import train_test_split
train_df, temp_df = train_test_split(df_res, train_size=0.8, random_state=42, stratify=df_res['category_encoded'])
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['category_encoded'])

tr_gen = ImageDataGenerator(rescale=1./255)
ts_gen = ImageDataGenerator(rescale=1./255)

train_gen = tr_gen.flow_from_dataframe(
    train_df, x_col='file_path', y_col='category_encoded',
    target_size=IMG_SIZE, class_mode='sparse', batch_size=BATCH_SIZE, shuffle=True
)
valid_gen = ts_gen.flow_from_dataframe(
    valid_df, x_col='file_path', y_col='category_encoded',
    target_size=IMG_SIZE, class_mode='sparse', batch_size=BATCH_SIZE, shuffle=True
)
test_gen = ts_gen.flow_from_dataframe(
    test_df, x_col='file_path', y_col='category_encoded',
    target_size=IMG_SIZE, class_mode='sparse', batch_size=BATCH_SIZE, shuffle=False
)

class DifferentialAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
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

custom_objects = {'DifferentialAttention': DifferentialAttention}
model = load_model(MODEL_PATH, custom_objects=custom_objects)

#freeze all layers, then unfreeze final layers
for layer in model.layers:
    layer.trainable = False

#unfreezing final layers for finetuning
total_layers = len(model.layers)
for layer in model.layers[total_layers-UNFREEZE_COUNT:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=FINE_TUNE_LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7),
    ModelCheckpoint(FINETUNE_PATH, monitor='val_loss', save_best_only=True)
]

model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks,
    verbose=1
)

loss, acc = model.evaluate(test_gen, verbose=1)
print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4%}")
model.save(FINETUNE_PATH)
print(f"Fine-tuned model saved to {FINETUNE_PATH}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

y_true = test_gen.classes

y_pred_probs = model.predict(test_gen, verbose=1)

y_pred = np.argmax(y_pred_probs, axis=1)

class_labels = list(test_gen.class_indices.keys())

report = classification_report(
    y_true,
    y_pred,
    target_names=class_labels,
    digits=4,
)
print("Classification Report:\n")
print(report)

from sklearn.utils.multiclass import unique_labels
import pandas as pd

report_dict = classification_report(
    y_true,
    y_pred,
    target_names=class_labels,
    output_dict=True
)
report_df = pd.DataFrame(report_dict).transpose()
print("\nClassification Report (as DataFrame):\n")
print(report_df)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

plt.figure(figsize=(8, 8))
disp.plot(cmap='Blues', xticks_rotation=45, values_format='d')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
