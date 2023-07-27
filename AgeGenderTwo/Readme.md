Real-Time Face Gender and Age Detection
=======================================


https://github.com/dxtaner/FaceImageClaassification/assets/44675799/08f8e9b2-3029-4986-b597-2317d1e3c076


Requirements
------------

*   Python (version 3.x)
*   OpenCV
*   cvlib
*   Keras (with TensorFlow backend)

Setup
-----

1.  Install the required libraries using pip:

    pip install opencv-python cvlib keras tensorflow

3.  Place the following model files in the same directory as the script:

*   GenderModel3.h5
*   ageModel1.h5

Usage
-----

1.  Connect a webcam to your computer.
2.  Run the Python script.
3.  A window will open, displaying the webcam feed with face detections.
4.  The script will draw bounding boxes around detected faces, along with predicted gender and approximate age range.

Output
------

The script will display the webcam feed with the following annotations for each detected face:

*   A red bounding box around the face.
*   The predicted gender (Erkek for male, Kadin for female) with its confidence score in percentage.
*   The predicted age group (0-20, 20-65, 65++) with its confidence score in percentage.





Termination
-----------

To stop the script, press the 'q' key while the webcam feed window is active.

_Note: For the best results, ensure sufficient lighting and a clear view of faces within the webcam frame._

Acknowledgments
---------------

The face detection model used in this script is provided by cvlib, and the gender and age prediction models are loaded using Keras with TensorFlow backend. Special thanks to the respective authors and contributors of these libraries.

_Disclaimer: The accuracy of the gender and age predictions may vary based on various factors like lighting conditions, image quality, and model performance. This script is intended for educational and demonstration purposes and may not be suitable for critical applications._
