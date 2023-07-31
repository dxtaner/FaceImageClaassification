Real-Time Face Gender and Age Prediction
========================================

This Python script uses OpenCV, Keras, and TensorFlow to perform real-time face gender and age prediction using a webcam. The script utilizes pre-trained deep learning models for gender and age classification.

Dependencies
------------

Make sure you have the following dependencies installed:

*   Python 3.x
*   OpenCV (cv2)
*   NumPy
*   Keras
*   TensorFlow

You can install the required packages using `pip`:

    pip install opencv-python numpy keras tensorflow

Usage
-----

1.  Clone this repository or download the Python script.
2.  Download the pre-trained model files `GenderModel3.h5` and `ageModel1.h5` and place them in the same directory as the script.
3.  Run the script using the following command:
    
        python face_gender_age_prediction.py
    
4.  The script will open your webcam and start detecting faces in real-time. The detected faces will be overlaid with predicted gender and age groups.
5.  Press 'q' to stop the script and close the webcam window.

Models
------

The script uses the following pre-trained models for gender and age classification:

*   Gender Model: `GenderModel3.h5`
*   Age Model: `ageModel1.h5`

Output
------

The script will display the webcam feed with bounding boxes around the detected faces, along with the predicted gender and age group labels.

The gender prediction is represented as "Erkek" (Male) or "Kadin" (Female).

The age prediction is represented as age ranges: "0-20", "20-65", or "65++".

Notes
-----

The face detection is performed using the Haar Cascade Classifier, which is available in OpenCV.

Ensure that your webcam is connected and functional before running the script.

Credits
-------

The models used in this script were trained on suitable datasets. Please check the original sources for more details.

License
-------

\[Include license information here, if applicable\]

Disclaimer
----------

This script is for educational and illustrative purposes only. The accuracy of gender and age prediction depends on the quality of the models and the input data. The script may not be perfectly accurate and should not be used for critical applications.
