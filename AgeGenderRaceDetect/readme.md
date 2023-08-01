
Real-Time Face Detection and Classification
===========================================

This Python script performs real-time face detection and classification using the cvlib, cv2, and keras libraries. It uses pre-trained models for gender, age, and race classification to analyze faces detected through the webcam.

Prerequisites
-------------

Before running the script, ensure you have the following installed:

*   Python
*   OpenCV (cvlib and cv2 libraries)
*   Keras (TensorFlow backend)
*   NumPy
*   Pafy (if using YouTube videos as input)

Installation
------------

1.  Clone or download this repository to your local machine.
2.  Make sure you have the required libraries installed using the following command:

    pip install cvlib opencv-python numpy tensorflow keras pafy

Usage
-----

Ensure your webcam is connected to the computer.

Run the Python script using the following command:

    python script_name.py

The webcam will start capturing live video, and any detected faces will be highlighted with a red rectangle.

The script will estimate face landmarks and use pre-trained models to classify gender, age, and race for each detected face.

The classification results will be displayed on the video feed, showing the predicted gender, age group, and race percentage.

Important Notes
---------------

*   For accurate results, ensure that the models raceModel1.h5, ageModel1.h5, and genderModel.h5 are present in the script's directory and loaded correctly.
*   The cvlib.detect\_face function is used for face detection, and it returns bounding boxes and confidence scores. Make sure the cvlib library is installed and functioning correctly.
*   The script is designed for real-time face analysis using the webcam. If you want to process video files, modify the script accordingly.

Acknowledgments
---------------

The face detection and classification tasks are made possible by using pre-trained models from Keras and TensorFlow.

Thanks to the developers of cvlib and cv2 libraries for simplifying face detection in Python.

The project is for educational purposes and can be extended to include more advanced classification tasks or different pre-trained models.

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.
