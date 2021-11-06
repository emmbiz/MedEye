

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

model_file_name = "mnet_shape_classifier_1636169917.h5"

# Load the trained model for prediction
model = models.load_model(model_file_name)

batch_size = 30
image_size = (224, 224)
train_folder = './train/'
test_folder = './test/'

# Get the class names
data_folder = 'dataset'
classes = os.listdir(data_folder)
classes.sort()
print("{} classes: {}".format(len(classes), classes))
label_map = {'normal': 0, 'covid': 1, 'not xray': 2}


print("Preparing testing dataset...")
def load_test_dataset(test_folder):
    # Store the test images and labels in a list
    images = []
    labels = []
    label_map = {'normal': 0, 'covid': 1, 'not xray': 2}
    for label in os.listdir(test_folder):
        label_path = os.path.join(test_folder, label)
        for img in os.listdir(label_path):
            img = os.path.join(label_path, img)
            img = image.load_img(img, target_size=image_size)
            img_array = image.img_to_array(img)
            img_batch = np.expand_dims(img_array, axis=0)
            #img_preprocessed = preprocess_input(img_batch)
            # Normalize the image
            #img_preprocessed = np.array(img_preprocessed, dtype="float") / 255.0
            img_preprocessed = np.array(img_batch, dtype="float") / 255.0

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
#results = imagenet_utils.decode_predictions(predictions)
#print("Results:", results)
indexes = np.argmax(predictions, axis=1)
print("Indexes:", indexes)
confidences = 100 * np.array(predictions)
print("Confidences:", confidences)
scores = tf.nn.softmax(predictions)
print("Scores:", scores)

# Plot results
import matplotlib.pyplot as plt
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
image_plt = test_images[i]
image_plt = np.squeeze(image_plt)
plot_image(predictions[i], test_labels[i], image_plt)
plt.subplot(1, 2, 2)
plot_value_array(predictions[i], test_labels[i])
plt.suptitle("Single Plot with bar graph")
plt.show()

print("This image most likely belongs to \'{}\' with a {:.2f} percent confidence."
            .format(class_names[np.argmax(predictions)], 100 * np.max(predictions)))

# Plot the first N test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 1
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2 * num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  #plot_image(predictions[i], label[i], images[i].numpy().astype("uint8"))imag = np.squeeze(imag)
  image_plt = test_images[i] #.astype("uint8")
  image_plt = np.squeeze(image_plt)
  plot_image(predictions[i], test_labels[i], image_plt)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  #plot_value_array(predictions[i], label[i])
  plot_value_array(predictions[i], test_labels[i])
plt.suptitle("Multiple plots")
plt.tight_layout()
plt.show()

def classify_image(image_path, image_label):
    img = image.load_img(image_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    # Normalize the image
    img_preprocessed = np.array(img_preprocessed, dtype="float") / 255.0

    prediction = model.predict(img_preprocessed)
    print("Prediction:", prediction)
    #result = imagenet_utils.decode_predictions(prediction)
    #print("Result:", result)
    index = np.argmax(prediction, axis=1)[0]
    print("Index:", index)
    confidence = 100 * np.array(prediction)
    print("Confidence:", confidence)
    score = tf.nn.softmax(predictions[0])
    print("Score:", score)

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(prediction[0], image_label, img)
    plt.subplot(1, 2, 2)
    plot_value_array(prediction[0], image_label)
    plt.suptitle("Single Plot with bar graph")
    plt.show()

print("Generating predictions from validation data...")
# Get the image and label arrays for the first batch of validation data
x_test = validation_generator[0][0]
y_test = validation_generator[0][1]

# Use the model to predict the class
class_probabilities = model.predict(x_test)

# The model returns a probability value for each class
# The one with the highest probability is the predicted class
predictions = np.argmax(class_probabilities, axis=1)

# The actual labels are hot encoded (e.g. [0 1 0], so get the one with the value 1
true_labels = np.argmax(y_test, axis=1)

# Use SciKit-Learn to plot a confusion matrix
from sklearn.metrics import confusion_matrix
# Plot the confusion matrix
cm = confusion_matrix(true_labels, predictions)
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=85)
plt.yticks(tick_marks, class_names)
plt.xlabel("Predicted Shape")
plt.ylabel("Actual Shape")
plt.show()


# Use SciKit-Learn to plot a confusion matrix
from sklearn.metrics import confusion_matrix
print("Generating predictions from validation data...")
print("\n**********************************************\n")
# Get the image and label arrays for the first batch of validation data
test_images_gen = validation_generator[0][0]
test_labels_gen = validation_generator[0][1]
#x_test = test_generator[0][0]
#y_test = test_generator[0][1]

# Use the model to predict the class
class_probabilities = model.predict(test_images_gen)
#class_probabilities = tf.squeeze(class_probabilities).numpy()

# The model returns a probability value for each class
# The one with the highest probability is the predicted class
prediction_digits = np.argmax(class_probabilities, axis=1)
# The actual labels are hot encoded (e.g. [0 1 0], so get the one with the value 1
true_labels = np.argmax(test_labels_gen, axis=1)

print(test_labels)


predicted = model.predict(test_images)
predicted_max = np.argmax(predicted, axis=1)

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
