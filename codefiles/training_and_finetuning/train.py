import os
import numpy as np 
import pandas as pd 
import seaborn as sns
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense, Dropout, 
                                     BatchNormalization, GaussianNoise, Input, Reshape)
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings("ignore")
sns.set_style('darkgrid')

dataset_path = "Train_Images"
data = []

for label in os.listdir(dataset_path):
    sub_dir = os.path.join(dataset_path, label)
    if os.path.isdir(sub_dir):
        for file_name in os.listdir(sub_dir):
            file_path = os.path.join(sub_dir, file_name)
            data.append([file_path, label])

df = pd.DataFrame(data, columns=['file_path', 'label'])
label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['label'])
df = df[['file_path', 'category_encoded']]

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(df[['file_path']], df['category_encoded'])
df_resampled = pd.DataFrame(X_resampled, columns=['file_path'])
df_resampled['category_encoded'] = y_resampled

df_resampled['category_encoded'] = df_resampled['category_encoded'].astype(str)

train_df_new, temp_df_new = train_test_split(
    df_resampled,
    train_size=0.8,
    shuffle=True,
    random_state=42,
    stratify=df_resampled['category_encoded']
)

valid_df_new, test_df_new = train_test_split(
    temp_df_new,
    test_size=0.5,
    shuffle=True,
    random_state=42,
    stratify=temp_df_new['category_encoded']
)

batch_size = 32
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)

tr_gen = ImageDataGenerator(rescale=1./255)
ts_gen = ImageDataGenerator(rescale=1./255)

train_gen_new = tr_gen.flow_from_dataframe(
    train_df_new,
    x_col='file_path',
    y_col='category_encoded',
    target_size=img_size,
    class_mode='sparse',
    color_mode='rgb',
    shuffle=True,
    batch_size=batch_size
)

valid_gen_new = ts_gen.flow_from_dataframe(
    valid_df_new,
    x_col='file_path',
    y_col='category_encoded',
    target_size=img_size,
    class_mode='sparse',
    color_mode='rgb',
    shuffle=True,
    batch_size=batch_size
)

test_gen_new = ts_gen.flow_from_dataframe(
    test_df_new,
    x_col='file_path',
    y_col='category_encoded',
    target_size=img_size,
    class_mode='sparse',
    color_mode='rgb',
    shuffle=False,
    batch_size=batch_size
)

class DifferentialAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim

    def build(self, input_shape):
        # input_shape = (batch, seq_len, channels)
        self.query_dense = tf.keras.layers.Dense(self.key_dim)
        self.key_dense   = tf.keras.layers.Dense(self.key_dim)
        self.value_dense = tf.keras.layers.Dense(self.key_dim)
        super().build(input_shape)

    def call(self, x):
        Q = self.query_dense(x)
        K = self.key_dense(x)
        V = self.value_dense(x)

        scores = tf.einsum('bij,bkj->bik', Q, K)  
        scores = tf.nn.softmax(scores, axis=-1)

        mask = tf.cast(scores > tf.reduce_mean(scores, axis=-1, keepdims=True), tf.float32)
        sparse_scores = scores * mask

        out = tf.einsum('bij,bjk->bik', sparse_scores, V)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim"  : self.key_dim,
        })
        return config

def create_inception_model(input_shape):
    inputs = Input(shape=input_shape)
    base_model = InceptionV3(weights='imagenet', input_tensor=inputs, include_top=False)
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    height, width, channels = 5, 5, 2048
    x = Reshape((height * width, channels))(x)
    attention_output = DifferentialAttention(num_heads=8, key_dim=channels)(x)
    attention_output = Reshape((height, width, channels))(attention_output)
    x = GaussianNoise(0.25)(attention_output)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = GaussianNoise(0.25)(x)
    x = Dropout(0.25)(x)
    outputs = Dense(5, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

model_path = "inception_attention_model_3.h5"
custom_objects = {'DifferentialAttention': DifferentialAttention}
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

input_shape = (224, 224, 3)

if os.path.exists(model_path):
    print("Loading existing model...")
    cnn_model = load_model(model_path, custom_objects=custom_objects)
else:
    print("Creating and training new model...")
    cnn_model = create_inception_model(input_shape)
    cnn_model.compile(optimizer=Adam(learning_rate=0.0001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    history = cnn_model.fit(
        train_gen_new,
        validation_data=valid_gen_new,
        epochs=20,
        callbacks=[early_stopping],
        verbose=1
    )
    cnn_model.save(model_path)
    print(f"Model saved to {model_path}")

test_labels = test_gen_new.classes
predictions = cnn_model.predict(test_gen_new)
predicted_classes = np.argmax(predictions, axis=1)
report = classification_report(test_labels, predicted_classes, target_names=list(test_gen_new.class_indices.keys()))
print(report)
conf_matrix = confusion_matrix(test_labels, predicted_classes)
print("Confusion Matrix:", conf_matrix)