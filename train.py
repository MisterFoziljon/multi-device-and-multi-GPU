import tensorflow as tf 
import numpy as np
import json
import os
import sys

def one_hot_encoder(class_number,label_size):
    label = np.zeros(label_size)
    label[class_number] = 1
    return label

def dataset_xyxy(batch_size):
    (x_train, y_train), _ = tf.keras.datasets.cifar100.load_data(label_mode="fine")
    x_train = x_train.astype('float32')/255.0    
    y_train = np.array([one_hot_encoder(class_number,100) for class_number in y_train])
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    return train_dataset

    
def compile_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'), 
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'), 
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'), 
        tf.keras.layers.MaxPooling2D(2, 2), 
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'), 
        tf.keras.layers.Flatten(), 
        tf.keras.layers.Dense(256, activation='relu'), 
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.Dense(256, activation='relu'), 
        tf.keras.layers.Dropout(0.3), 
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.Dense(100, activation='softmax')])

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
        optimizer=tf.keras.optimizers.Adam(), 
        metrics=tf.keras.metrics.CategoricalAccuracy()) 

    return model

tf_config = {"cluster": {"worker": ["192.169.0.146:12345", "192.169.0.128:12345"]}, 
             "task": {"index": 0, "type": "worker"}}

num_workers = len(tf_config['cluster']['worker'])

print(f"Number workers: {num_workers}")

per_worker_batch_size = 256
global_batch_size = per_worker_batch_size * num_workers

strategy = tf.distribute.MultiWorkerMirroredStrategy()
dataset = dataset_xyxy(global_batch_size)

with strategy.scope():
    model = compile_model()

print("Train boshlandi")
model.fit(dataset, epochs=10, steps_per_epoch=10)
model.save("multi_model.h5")
