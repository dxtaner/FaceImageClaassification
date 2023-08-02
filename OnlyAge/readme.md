Age Detection with OpenCV and Keras
===================================

This project demonstrates age detection using OpenCV and Keras. It utilizes a pre-trained deep learning model to estimate the age of faces in real-time video from a YouTube URL.

Table of Contents
-----------------

*   [Overview](#overview)
*   [Requirements](#requirements)
*   [Installation](#installation)
*   [Usage](#usage)
*   [Credits](#credits)
*   [License](#license)

Overview
--------

This project uses OpenCV and Keras to detect faces in a real-time video stream from a YouTube URL and estimates the age of each detected face. The age estimation model is trained on various age groups and can classify a person's age into five categories: 0-18, 18-30, 30-45, 45-65, and 65++.

Requirements
------------

*   Python 3.x
*   OpenCV
*   cvlib
*   numpy
*   pafy
*   Keras
*   TensorFlow

Installation
------------

1.  Clone the repository:

    git clone https://github.com/your_username/age_detection_project.git
    cd age_detection_project

2.  Install the required dependencies:

    pip install opencv-python cvlib numpy pafy keras tensorflow

3.  Download the pre-trained age estimation model `ageModel5li.h5` and place it in the project root directory.

Usage
-----

1.  Open the `detectage.py` script and replace the `url` variable with the YouTube URL of the video you want to analyze for age detection.
2.  Run the age detection script:

    python detectage.py

3.  The script will open a window showing the real-time video stream with age estimates displayed on each detected face.
4.  Press 'q' to quit the video stream.

Credits
-------

The age estimation model used in this project is provided by <insert source here> (if applicable).

License
-------

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
