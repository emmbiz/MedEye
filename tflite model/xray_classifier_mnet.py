
# import required packages
import tensorflow as tf
from keras.applications import mobilenet
from keras.applications.mobilenet_v2 import preprocess_input
from tensorflow import keras
import tensorflow_hub as hub
from keras import models
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, image

import os
import time
import glob
import shutil
import numpy as np

# Get the class names
data_folder = 'dataset'
classes = os.listdir(data_folder)
classes.sort()
print("{} classes: {}".format(len(classes), classes))

# Prepare datasets for training and testing
train_folder = './train/'
test_folder = './test/'
for class_name in classes:
  img_path = os.path.join(data_folder, class_name)
  images = glob.glob(img_path + '/*.jpg')
  images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
  print("{} images for '{}'".format(len(images), class_name))

  # Divide data for training and testing
  train_num = int(round(len(images) * 0.87))
  test_num = len(images) - train_num
  print("{} training => {} images".format(class_name, train_num))
  print("{} testing => {} images".format(class_name, test_num))
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
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        #fill_mode='nearest',
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

print("map:", train_generator.class_indices)
label_map = train_generator.class_indices
class_names = list(train_generator.class_indices.keys())
print("Training Data Size:", train_generator.samples)
print("Data generators ready.")

# Plot the 9 images from the training dataset
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.subplots_adjust(wspace=0.3)
    plt.subplots_adjust(hspace=0.3)
    plt.imshow(train_generator[0][0][i])
    class_index = np.argmax(train_generator[0][1][i])
    plt.title(class_names[class_index])
plt.suptitle("Training Sample")
plt.show()

# Plot transformed training images
import matplotlib.pyplot as plt
def plot_images(images_arr):
    images = images_arr
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for image, ax in zip(images, axes):
        ax.imshow(image)
    plt.suptitle("Image Data Transformations")
    plt.tight_layout()
    plt.show()
augmented_imgs = [train_generator[0][0][0] for i in range(5)]
plot_images(augmented_imgs)

# Define the CNN classifier network
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

# Transfer Learning with MobileNet
base_model = keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

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
model_weights_name = "mnet_classifier_weights_{}.h5".format(int(tm))
export_path_wgs = "./" + model_weights_name
model.save_weights("mnet_classifier_weights.h5")

# Save the model and create h5 and tflite file format
model_file_name = "mnet_xray_classifier_{}.h5".format(int(tm))
export_path_mdl = "./" + model_file_name
model.save(export_path_mdl)
#tf.keras.models.save_model(model, model_file_name)
#tf.saved_model.save(model, "/mod/")
print("Model saved as", model_file_name)

# Convert Keras model to TF Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter = lite.TFLiteConverter.from_keras_model(model)
tflite_float_model = converter.convert()
open("xray_classifier.tflite", "wb").write(tflite_float_model)

# Show model size in KBs
float_model_size = len(tflite_float_model) / 1024
print("TFLite Float model size = %d KBs." % float_model_size)

print("Preparing testing dataset...")
def load_test_dataset(test_folder):
    # Store the test images and labels in a list
    images = []
    labels = []
    for label in os.listdir(test_folder):
        label_path = os.path.join(test_folder, label)
        for img in os.listdir(label_path):
            img = os.path.join(label_path, img)
            img = image.load_img(img, target_size=image_size)
            img_array = image.img_to_array(img)
            img_batch = np.expand_dims(img_array, axis=0)
            img_preprocessed = preprocess_input(img_batch)
            # Normalize the image
            img_preprocessed = np.array(img_preprocessed, dtype="float") / 255.0
            #img_preprocessed = np.array(img_batch, dtype="float") / 255.0

            images.append(img_preprocessed)
            labels.append(label_map[label])

    # images = np.array(all_images)  # Create batch
    # labels = np.array(all_labels)  # Create batch

    # Stack up images list to pass for prediction
    #stacked = np.vstack(images)
    return images, labels

test_images, test_labels = load_test_dataset(test_folder)

#predictions = model.predict(np.vstack(test_images), batch_size=3)
predictions = model.predict(np.vstack(test_images))
print("Predictions:", predictions)
indexes = np.argmax(predictions, axis=1)
print("Indexes:", indexes)
confidences = 100 * np.array(predictions)
print("Confidences:", confidences)
scores = tf.nn.softmax(predictions)
print("Scores:", scores)

# Plot results
def plot_image(predictions_array, true_label, img):

    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)

    color = 'blue' if predicted_label == true_label else 'red'

    #plt.title(class_names[label[i]], color=color)
    plt.xlabel("{} {:2.0f}% [{}]".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)

def plot_value_array(predictions_array, true_label):

    plt.xticks(range(len(class_names)))
    plt.yticks([])
    plt.grid(False)

    thisplot = plt.bar(range(len(class_names)), predictions_array, color="#777777")
    plt.ylim([0, 1])

    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
    #thisplot[predictions_array].set_color('blue')

print("Predictions")
print(predictions[0])
print(test_labels[0])
print(test_images[0])

# Plot a single image
print("Plotting Single Plot...")
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
#path = circle_image = './test/circle/circle321.jpg'
#img = image.load_img(path, target_size=image_size)
image_plt = test_images[i] #.astype("uint8")
image_plt = np.squeeze(image_plt)
plot_image(predictions[i], test_labels[i], image_plt)
plt.subplot(1, 2, 2)
plot_value_array(predictions[i], test_labels[i])
plt.suptitle("Single Plot Result")
plt.show()

index = np.argmax(predictions, axis=1)[0]

print("This image most likely belongs to \'{}\' with a {:.2f} percent confidence."
            .format(class_names[index], 100 * np.max(predictions)))

# Plot the first N test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 1
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2 * num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  image_plt = np.squeeze(test_images[i])
  plot_image(predictions[i], test_labels[i], image_plt)

  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(predictions[i], test_labels[i])
plt.suptitle("Multiple Plot Results")
plt.tight_layout()
plt.show()

print("Generating predictions from validation data...")
# Get the image and label arrays for the first batch of validation data
images_array = validation_generator[0][0]
labels_array = validation_generator[0][1]
# Use the model to predict the class
class_probabilities = model.predict(images_array)
# Get the class with the highest probability
predictions = np.argmax(class_probabilities, axis=1)
# The actual labels are hot encoded (e.g. [0 1 0], so get the one with the value 1
true_labels = np.argmax(labels_array, axis=1)

# Use SciKit-Learn to plot a confusion matrix
from sklearn.metrics import confusion_matrix
# Plot the confusion matrix
cm = confusion_matrix(true_labels, predictions)
#cm = confusion_matrix(test_labels, indexes)
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=85)
plt.yticks(tick_marks, class_names)
plt.xlabel("Predicted Shape")
plt.ylabel("Actual Shape")
plt.show()

# Plot Training Loss and Accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
