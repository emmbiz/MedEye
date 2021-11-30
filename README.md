# Introduction
MedEye is an image classification app that can classify chest X-ray images to predict the likelihood that an X-ray image belongs to a person infected with Covid-19 or not. It bypasses the process of using testing kits by processing X-ray images captured through the device camera to detect the presence of Covid-19 from the given image.


# Inspiration
Early diagnosis of Covid-19 could be difficult and people who are already infected may not be fully aware so as to begin necessary treatment as soon as possible. Accelerating diagnosis of Covid-19 for faster treatment time and quicker eradication of the virus is therefore an important public health concern.

Given the limited number of Covid-19 testing kits available for wide-range use, this application was developed to solve the problem of relying on only one measure of diagnosis.

Since Covid-19 attacks the epithelial cells that line our respiratory tract, X-ray images can be used to analyze the health of an individual’s lungs to determine the likelihood that the individual has been infected with Covid-19.

Hence, this solution can be used by anyone to enable early detection and faster treatment time.


# How it was built
- X-ray images belonging to patients diagnosed with Covid-19, and normal X-ray images were gotten from a Kaggle dataset and separated into categories for training and testing.
- Training images were further divided into training and validation sets to train the model with the training set and validate the performance of the training process with the validation set.
- A CNN classifier network was trained using Transfer Learning on the MobileNetV2 network, and a Keras model was developed and evaluated for best predictive performance.
- The trained model was converted to TensorFlow Lite format and used in an Android application developed in Kotlin making use of TensorFlow Lite libraries and other application dependencies.
- The developed Android application was tested to assess the practical performance of the model, and an iterative model training process was conducted to improve the model’s performance and to validate its functionality.
- The final application utilizing the best trained model was built and generated as a signed .apk for installation and use on an Android device.


# Challenges faced
Classifying X-ray images only without classifying non-X-ray images was initially difficult due to training on only two possible classes. However, this was overcome by finding a dataset containing negative images and retraining the model with these negative images to predict a class for images that are not X-ray images, or cannot be classified as such.


# Accomplishment
Built an Android app that can classify X-ray images to predict the presence of Covid-19 in the body with good accuracy.


# Kaggle Datasets
- Chest X-Ray Images: https://www.kaggle.com/ankitachoudhury01/covid-patients-chest-xray
- Negative Images: https://www.kaggle.com/muhammadkhalid/negative-images


