import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
import numpy as np
import matplotlib.pyplot as plt
import os

def get_next_train_folder(base_path, train_prefix):
    items = os.listdir(base_path)
    train_folders = [item for item in items if os.path.isdir(os.path.join(base_path, item)) and item.startswith(train_prefix)]
    max_num = max([int(folder.split('_')[-1]) for folder in train_folders], default=0)
    return f"{train_prefix}_{max_num + 1}"

graphs_base_path = 'graphs'
models_base_path = 'models'
train_prefix = 'CNN1_model_train'
next_graph_folder = get_next_train_folder(graphs_base_path, train_prefix)
next_model_folder = get_next_train_folder(models_base_path, train_prefix)

next_graph_path = os.path.join(graphs_base_path, next_graph_folder)
next_model_path = os.path.join(models_base_path, next_model_folder)
os.makedirs(next_graph_path, exist_ok=True)
os.makedirs(next_model_path, exist_ok=True)

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_dir = 'train_dataset'

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),  
    batch_size=32,
    class_mode='categorical'  
)

val_dir = 'val_dataset'

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(64, 64),  
    batch_size=32,
    class_mode='categorical'  
)

# Build the CNN model
def build_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(64, 64, 3)),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

num_classes = len(train_generator.class_indices)  
model = build_model(num_classes)

history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator  
)

model.save(os.path.join(next_model_path, 'CNN1_model.h5'))

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(os.path.join(next_graph_path, 'CNN1_accuracy.png'))
plt.show()


plt.plot(history.history['loss'])
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(os.path.join(next_graph_path, 'CNN1_loss.png'))
plt.show()


test_dir = 'test_dataset'
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),  
    batch_size=32,
    class_mode='categorical',
    shuffle=False  
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy:.2f}, Test loss: {test_loss:.2f}")