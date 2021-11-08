# Introduction
MedEye is a real-time image classifier application that can classify chest X-ray images to predict whether an X-ray image belongs to a person infected with Covid-19 or not. It bypasses the use of testing kits by processing X-ray images in real-time through the device camera to detect the presence of Covid-19 in a given X-ray image.


# Inspiration
Early diagnosis of Covid-19 could be difficult, and people with mild symptoms may not be fully aware so as to begin necessary treatment as soon as possible. Accelerating diagnosis of Covid-19 for faster time to treatment is therefore an important public health concern.
Given the limited number of Covid-19 testing kits available for Covid-19 testing, this application was developed to solve the problem of relying on only one measure of diagnosis.
Since Covid-19 attacks the epithelial cells that line our respiratory tract, X-ray images can be used to analyze the health of an individual’s lungs to determine the likelihood of being infected with Covid-19.
Hence, this solution was created to be used by anyone to enable early detection and faster treatment time.


# How it was built
- X-ray images belonging to persons diagnosed with Covid-19 and normal X-ray images were gotten from the same Kaggle dataset, and then separated into categories for training, testing, and validation.
- A Keras model was developed through training with images from the training set and validation set to validate training procedure, and then tested with images from the testing set.
- The model was exported as a TensorFlow Lite model and downloaded as a .zip file with the contents extracted and placed in the Android project’s assets folder.
- The Android application was developed in Kotlin making use of TensorFlow Lite libraries and other application dependencies.
- The developed application was tested to assess the performance of the model, and an iterative model training process was conducted to improve the model performance and to validate its core functionality.
- The final application using the best model iteration was generated as a signed .apk in order to be installed and used on an Android device


# Screenshots  

| Normal      | Covid      |
|------------|-------------|
| ![Screenshot 1_normal - Copy](https://user-images.githubusercontent.com/87937713/131337247-477a6d89-cdd6-4bc1-a259-0086a3d8000d.png) | ![Screenshot 2_covid - Copy](https://user-images.githubusercontent.com/87937713/131337255-da4f1fa7-57bc-4409-84db-98284f1733a5.png) |


# Challenges faced
Due to using a real-time image classifier through the device camera, it was initially difficult to classify X-ray images only without the model trying to classify unknown images and backgrounds.
This problem was solved by adding more negative images (images that don't belong to either class) to the classifier to improve its accuracy, and then retraining the model and importing the contents into the Android Studio project.


# Accomplishment
Built an Android app that can classify X-ray images to predict the likelihood of Covid-19 being present in the body of its owner with a good accuracy.


