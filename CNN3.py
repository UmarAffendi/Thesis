import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, Activation, Dropout, BatchNormalization, Conv1D,Conv2D, MaxPooling1D,MaxPooling2D,GlobalAveragePooling1D,GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.layers import Input, Flatten
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import initializers
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler


def get_next_train_folder(base_path, train_prefix):
    items = os.listdir(base_path)
    train_folders = [item for item in items if os.path.isdir(os.path.join(base_path, item)) and item.startswith(train_prefix)]
    max_num = max([int(folder.split('_')[-1]) for folder in train_folders], default=0)
    return f"{train_prefix}_{max_num + 1}"

graphs_base_path = 'graphs'
models_base_path = 'models'
train_prefix = 'CNNcomp_model_train'
next_graph_folder = get_next_train_folder(graphs_base_path, train_prefix)
next_model_folder = get_next_train_folder(models_base_path, train_prefix)

next_graph_path = os.path.join(graphs_base_path, next_graph_folder)
next_model_path = os.path.join(models_base_path, next_model_folder)
os.makedirs(next_graph_path, exist_ok=True)
os.makedirs(next_model_path, exist_ok=True)

datagen = ImageDataGenerator(
    rescale=1./255,
)

train_dir = 'improved_dataset\\train'

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),  
    batch_size=100,
    class_mode='categorical'  
)

val_dir = 'improved_dataset\\val'

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(64, 64),  
    batch_size=100,
    class_mode='categorical'  
)

ac_fun = 'relu'
weight_initializer = initializers.GlorotNormal()


def build_model(num_classes):
    

    
    model = tf.keras.Sequential([

        # tf.keras.layers.InputLayer(input_shape=(64, 64, 3)),

        # tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(512, activation='relu'),
        # tf.keras.layers.Dropout(0.5),
        # tf.keras.layers.Dense(num_classes, activation='softmax')

        tf.keras.layers.InputLayer(input_shape=(64, 64, 3)),

        tf.keras.layers.Conv2D(32, (3, 3), activation=ac_fun, padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(32, (3, 3), activation=ac_fun, padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(64, (3, 3), activation=ac_fun, padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(64, (3, 3), activation=ac_fun, padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(128, (3, 3), activation=ac_fun, padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(1, 1)),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(128, (3, 3), activation=ac_fun, padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(1, 1)),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(256, (3, 3), activation=ac_fun, padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(1, 1)),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')


    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

    # inputs = Input(shape=(64,64,3))
    # x = inputs
    # x = Conv2D(16,(3,3),strides = (1,1),padding='valid',activation = ac_fun,kernel_initializer=weight_initializer)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = Conv2D(16,(3,3),strides = (1,1),padding='valid',activation = ac_fun,kernel_initializer=weight_initializer)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = Conv2D(16,(3,3),strides = (2,2),padding='valid',activation = ac_fun,kernel_initializer=weight_initializer)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = Conv2D(16,(3,3),strides = (1,1),padding='valid',activation = ac_fun,kernel_initializer=weight_initializer)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = Conv2D(16,(3,3),strides = (2,2),padding='valid',activation = ac_fun,kernel_initializer=weight_initializer)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = Conv2D(16,(3,3),strides = (1,1),padding='valid',activation = ac_fun,kernel_initializer=weight_initializer)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = Conv2D(16,(3,3),strides = (2,2),padding='valid',activation = ac_fun,kernel_initializer=weight_initializer)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # x = Flatten()(x)
    # x = Dropout(0.3)(x)
    # #x = Dense(2048,activation = ac_fun,kernel_initializer=weight_initializer)(x)
    # x = Dense(512,activation = ac_fun,kernel_initializer=weight_initializer)(x)
    # x = Dense(128,activation = ac_fun,kernel_initializer=weight_initializer)(x)
    
    # outputs = Dense(num_classes,activation=None,kernel_initializer=weight_initializer)(x)
    # model = Model(inputs = inputs, outputs = outputs)
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # return model

num_classes = len(train_generator.class_indices)  
model = build_model(num_classes)

history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator  
)

model.save(os.path.join(next_model_path, 'CNNcomp_model.h5'))

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(os.path.join(next_graph_path, 'CNNcomp_accuracy.png'))
plt.show()


plt.plot(history.history['loss'])
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(os.path.join(next_graph_path, 'CNNcomp_loss.png'))
plt.show()


test_dir = 'test_dataset'
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),  
    batch_size=100,
    class_mode='categorical',
    shuffle=False  
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy:.2f}, Test loss: {test_loss:.2f}")