"""
Transfer learning on MobileNetV2 for defect detection, quantized mapping to INT8.
"""
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Image size for TFLite Micro standard examples (96x96 grayscale/RGB)
IMG_SHAPE = (96, 96, 3)

base_model = MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(32, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# example code
# model.fit(train_ds, validation_data=val_ds, epochs=10)

def representative_dataset():
    for _ in range(100):
      data = tf.random.uniform([1, 96, 96, 3], 0, 255, dtype=tf.float32)
      yield [data]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()
with open('defect_sorter_quant.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print("Exported INT8 Model successfully.")
