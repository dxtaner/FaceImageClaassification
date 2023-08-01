Real-time Gender and Age Detection in Video Stream
==================================================

This Python script performs real-time gender and age detection on faces in a video stream from a YouTube URL using pre-trained deep learning models. It utilizes OpenCV, cvlib, Keras, and TensorFlow to achieve this functionality.

Requirements
------------

Before running the script, ensure you have the following dependencies installed:

*   Python (version >= 3.6)
*   OpenCV (cv2)
*   cvlib
*   Keras
*   TensorFlow
*   numpy
*   pafy

You can install the required packages using pip:

pip install opencv-python cvlib keras tensorflow numpy pafy

Pre-trained Models
------------------

The script uses pre-trained deep learning models for gender and age detection, which are loaded from the following files:

*   Gender Model: GenderModel3.h5
*   Age Model: AgeModel3.h5

You can download these models or use your own trained models for gender and age classification.

Usage
-----

1.  Clone or download the repository containing the script and the pre-trained models.
2.  Replace the GenderModel3.h5 and AgeModel3.h5 files with your own trained models if desired. Make sure the model architectures and labels match the ones used in the script.
3.  Execute the script using Python:

python detectvideo.py

The script will prompt you to enter a YouTube URL from which it will stream the video and perform real-time gender and age detection on detected faces.

The script will open a new window showing the video stream with gender and age labels overlaid on the detected faces.

Press 'q' on the keyboard to exit the script and close the video stream window.

**Note:** The accuracy of gender and age prediction depends on the quality and diversity of training data used for the models. The provided models and the script's performance can be further improved by fine-tuning the models on custom datasets.

Acknowledgments
---------------

*   The gender and age detection models are based on deep learning models trained on labeled datasets.
*   The face detection in the script is performed using the cvlib.detect\_face function, which leverages OpenCV's Haar cascades.
*   The real-time video streaming from YouTube is made possible by the pafy library, which allows us to extract video URLs from YouTube.

License
-------

This project is licensed under the MIT License. Feel free to use and modify it according to your needs.
