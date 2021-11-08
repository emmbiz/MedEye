# Import required packages
import tensorflow as tf
from keras import models
from keras.preprocessing.image import image

import os
import numpy as np

model_file_name = "mnet_xray_classifier.h5"

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
label_map = {'covid': 0, 'normal': 1, 'not xray': 2}
class_names = list(label_map.keys())
print("CLASS", class_names)

# Plot the results
import matplotlib.pyplot as plt
def plot_image(predictions_array, true_label, img):
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'

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

def classify_image(image_path, image_label):
    img = image.load_img(image_path, target_size=image_size)
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    # Normalize the image
    img_preprocessed = np.array(img_batch, dtype="float") / 255.0

    # Use the model to predict the class
    prediction = model.predict(img_preprocessed)

    print("Prediction:", prediction)
    index = np.argmax(prediction, axis=1)[0]
    print("Index:", index)
    confidence = 100 * np.array(prediction)
    print("Confidence:", confidence)
    score = tf.nn.softmax(prediction[0])
    print("Score:", score)

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(prediction[0], image_label, img)
    plt.subplot(1, 2, 2)
    plot_value_array(prediction[0], image_label)
    plt.suptitle("Single Plot with bar graph")
    plt.show()

    print("This image most likely belongs to \'{}\' with a {:.2f} percent confidence."
          .format(class_names[index], 100 * np.max(prediction)))

for label in os.listdir(test_folder):
    label_path = os.path.join(test_folder, label)
    for img in os.listdir(label_path):
        img = os.path.join(label_path, img)

        classify_image(img, label_map[label])
