# Gender and Age Detection from Video

This project utilizes computer vision and deep learning models to detect and classify gender and age from a video stream. The Python script uses OpenCV, cvlib, and pre-trained Keras models for gender and age classification. The detected faces are boxed with rectangles, and the predicted gender and age labels are displayed above the respective faces in real-time.


## Prerequisites

Before running the script, ensure you have the following prerequisites installed:

- Python (Python 3.6 or later)
- OpenCV library (`pip install opencv-python`)
- cvlib library (`pip install cvlib`)
- Numpy library (`pip install numpy`)
- Keras library (`pip install keras`)
- TensorFlow library (`pip install tensorflow`)
- pafy library (`pip install pafy`)

## Model Files

Place the following model files in the same directory as the script:

- `GenderModel3.h5`: Pre-trained Keras model for gender classification.
- `ageModel1.h5`: Pre-trained Keras model for age classification.

## Usage

1. Replace the `url` variable with the desired YouTube video URL in the script.
2. Run the Python script. It will capture the video stream from the specified URL and process it in real-time.
3. The script will draw rectangles around detected faces and display the predicted gender and age labels above the respective faces.

**Note:** The accuracy of predictions depends on the quality and resolution of the video stream and the performance of the pre-trained models.

## Important Note

This script uses pre-trained models for gender and age classification, which might not be perfect and may produce incorrect predictions in certain cases. It's essential to review and comply with the terms and conditions of the libraries and models used in this project.

Always consider privacy and legal considerations when working with computer vision technologies, and ensure you have the necessary rights to use the models and libraries.

## Acknowledgments

- cvlib: [https://github.com/arunponnusamy/cvlib](https://github.com/arunponnusamy/cvlib)
- pafy: [https://github.com/mps-youtube/pafy](https://github.com/mps-youtube/pafy)

Make sure to acknowledge and give credit to the creators of cvlib and pafy, as well as the authors of the pre-trained models used in this project.

**Disclaimer**: This project may use certain pre-trained models and libraries that could be subject to license and usage restrictions. Always comply with the terms and conditions of the libraries and models used in your project.
