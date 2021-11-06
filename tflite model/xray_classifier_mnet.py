# import required packages
import tensorflow as tf
from keras.applications import mobilenet
from keras.applications.mobilenet_v2 import preprocess_input
from tensorflow import keras
import tensorflow_hub as hub
from keras import models
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, image
from tensorflow.keras.applications import imagenet_utils

from tensorflow.keras import applications
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
from tensorflow.keras import optimizers
import PIL
import os
import time
import glob
import shutil
import numpy as np

print('Pillow Version:', PIL.__version__)
print('TensorFlow version:', tf.__version__)
print('Keras version:', keras.__version__)

# Get the class names
data_folder = 'dataset'
classes = os.listdir(data_folder)
classes.sort()
print("{} classes: {}".format(len(classes), classes))
label_map = {'circle': 0, 'square': 1, 'triangle': 2}

# Prepare datasets for training and testing
train_folder = './train/'
test_folder = './test/'
for class_name in classes:
  img_path = os.path.join(data_folder, class_name)
  images = glob.glob(img_path + '/*.jpg')
  images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
  print("{} images for '{}'".format(len(images), class_name))

  # Divide data into training and testing (80%-20%)
  train_num = int(round(len(images) * 0.80))
  print("Training => {} images".format(train_num))
  training_set, testing_set = images[:train_num], images[train_num:]

  for train in training_set:
    if not os.path.exists(os.path.join(train_folder, class_name)):
      os.makedirs(os.path.join(train_folder, class_name))
    shutil.copy(train, os.path.join(train_folder, class_name))

  for test in testing_set:
    if not os.path.exists(os.path.join(test_folder, class_name)):
      os.makedirs(os.path.join(test_folder, class_name))
    shutil.copy(test, os.path.join(test_folder, class_name))

# Generate batches of image data (and their labels) from training folder
print("Getting Data...")
train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2)

batch_size = 30
image_size = (224, 224)
print("Preparing training dataset...")
train_generator = train_datagen.flow_from_directory(
    directory=train_folder,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

print("Preparing validation dataset...")
validation_generator = train_datagen.flow_from_directory(
    directory=train_folder,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

class_names = list(train_generator.class_indices.keys())
print("Classes:", class_names)
print("Size:", train_generator.samples)
print("Data generators ready.")

# Plot the first 9 images from the training dataset
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_generator[0][0][i])
    class_index = np.argmax(train_generator[0][1][i])  # [i]
    #plt.xlabel(class_names[class_index])
    plt.title(class_names[class_index])
    #plt.axis("off")
plt.suptitle("Training Dataset Sample")
plt.show()

# Plot transformed training images
import matplotlib.pyplot as plt
def plot_images(images_arr):
    images = images_arr
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for image, ax in zip(images, axes):
        ax.imshow(image)
    plt.suptitle("Image Transformations")
    plt.tight_layout()
    plt.show()
augmented_imgs = [train_generator[0][0][0] for i in range(5)]
plot_images(augmented_imgs)

# Define the CNN classifier network
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

# Use Transfer Learning with MobileNet
# MobileNet
base_model = keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#rsnet_model = keras.applications.resnet.ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Add custom last layer
'''
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
IMAGE_RES = 224
feature_extractor = hub.KerasLayer(URL, input_shape=(IMAGE_RES, IMAGE_RES, 3), trainable=False)
model = tf.keras.Sequential([
  feature_extractor,
  tf.keras.layers.Dense(len(class_names), activation='softmax')
])'''

# Freeze the already-trained layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Create prediction layer for classification of our images
x = base_model.output
x = Flatten()(x)
prediction_layer = Dense(len(class_names), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=prediction_layer)

# Compile the model for categorical (multi-class) classification
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
'''
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])'''

# Now print the full model, which will include the layers of the base model plus the dense layer we added
print(model.summary())

# Train the model including the validation holdout dataset for validation
EPOCHS = 5
history = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size,
    epochs = EPOCHS)

# Save the trained model weights and model
tm = time.time()
model_weights_name = "shape_weights_{}.h5".format(int(tm))
export_path_wgs = "./" + model_weights_name
model.save_weights("mnet_shape_weights.h5")

# Save the model and create h5 and tflite file format
model_file_name = "mnet_shape_classifier_{}.h5".format(int(tm))
export_path_mdl = "./" + model_file_name
model.save(export_path_mdl)
#tf.keras.models.save_model(model, model_file_name)
#tf.saved_model.save(model, "/mod/")
print("Model saved as", model_file_name)

# Convert Keras model to TF Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter = lite.TFLiteConverter.from_keras_model(model)
tflite_float_model = converter.convert()
open("shapes_classifier.tflite", "wb").write(tflite_float_model)

# Show model size in KBs
float_model_size = len(tflite_float_model) / 1024
print("TFLite Float model size = %d KBs." % float_model_size)