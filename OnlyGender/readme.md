
Face and Gender Detection from YouTube Video
============================================

This Python script allows you to detect faces in a YouTube video stream and predict the gender of the detected faces using a pre-trained gender classification model.

Prerequisites
-------------

Before running the script, make sure you have the following installed:

*   Python (>= 3.6)
*   OpenCV (cvlib and cv2)
*   NumPy
*   pafy
*   TensorFlow (>= 2.x) with Keras API

Installation
------------

1.  Clone the repository:

    git clone [https://github.com/your_username/your_repository.git](https://github.com/dxtaner/FaceImageClaassification/new/main/OnlyGender)

3.  Change directory to the project folder:

    cd your_repository

5.  Install the required dependencies:

    pip install cvlib cv2 numpy pafy tensorflow keras

Usage
-----

1.  Replace `../genderModel.h5` with the path to your pre-trained gender classification model in the `load_model` function:

    gender_model = load_model("path/to/your/genderModel.h5")

3.  Replace the `url` variable with the YouTube video URL you want to analyze:

    url = 'https://www.youtube.com/watch?v=your_youtube_video_id'

5.  Run the script:

    python detectgender.py

Output
------

The script will open a window showing the YouTube video stream with faces bounded by rectangles and the predicted gender of each detected face displayed above the rectangle.

Notes
-----

*   The `cvlib` library is used to detect faces in the video stream.
*   The `genderModel.h5` is a pre-trained gender classification model that predicts the gender of the detected faces as either "Erkek" (male) or "Kadin" (female).

Acknowledgments
---------------

*   The face detection is powered by `cvlib`, a computer vision library based on OpenCV.
*   The gender classification model is trained using TensorFlow and Keras.

License
-------

\[Specify your license here\]

Feel free to modify this template as per your requirements and add any additional information that might be relevant to your project.
