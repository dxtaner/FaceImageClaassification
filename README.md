# FaceImageClaassification

![logoWall2](https://user-images.githubusercontent.com/44675799/145050435-80ccfb4d-9e76-41b9-b470-4f1b45c73609.jpg)

## Age, Gender, and Race Classification with Convolutional Neural Networks

### Dataset

UTKFace dataset is a large-scale face dataset with long age span (range from 0 to 116 years old). The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc. This dataset could be used on a variety of tasks, e.g., face detection, age estimation, age progression/regression, landmark localization, etc. Some sample images are shown as following

Link: https://www.kaggle.com/datasets/jangedoo/utkface-new
Link: https://susanqq.github.io/UTKFace/

![image](https://user-images.githubusercontent.com/44675799/145047648-70b091a7-233c-446d-835b-cac733b2f9a0.png)

## Distribution of Gender, Age, and Race in the dataset

## Gender graphic
![image](https://user-images.githubusercontent.com/44675799/145048886-a7c335c8-5cd9-49b2-951f-5ad33d2b445c.png)
## Age graphic
![image](https://user-images.githubusercontent.com/44675799/145048976-a0f987a3-c3ba-4c30-8b4a-1f845715a4bf.png)

## Label

The labels of each face image is embedded in the file name, formated like [age]_[gender]_[race]_[date&time].jpg
<pre>[age] is an integer from 0 to 116, indicating the age
[gender] is either 0 (male) or 1 (female)
[race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
[date&time] is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace </pre>

## Some Layers

### Max-pooling

When the training dataset is not large enough to contain all the features in the whole dataset, overfitting happens. By adding max-pooling layer, the size of spatial size and the number of parameters will be reduced (et. only a subset of features which has the max value will be selected), as a result the model is less likely to learn false patterns.
### Dropout

One of the major challenges in training Deep Neural Network is deciding when to stop the training. With too short time of training, underfitting occurs, while with too long time of training, overfitting occurs. This project will also apply early stopping to reduce overfitting – stop training when performance on the validation dataset starts to degrade

# Result 

## Gender 
![image](https://user-images.githubusercontent.com/44675799/145049117-07b6e328-f2a5-4b0e-bd67-157d69cb45fd.png)
![image](https://user-images.githubusercontent.com/44675799/145049152-a113f350-6a6c-437d-ae71-22fcd85134bb.png)
![image](https://user-images.githubusercontent.com/44675799/145049130-e4fba651-2b91-4b2e-8938-fb365b52565d.png)
## Age
![image](https://user-images.githubusercontent.com/44675799/145049187-0a24a0c0-e41f-4dc2-a357-b6bb7c5c5637.png)
![image](https://user-images.githubusercontent.com/44675799/145049180-e20116a4-d7ba-428e-8ce6-5f4aa55e3dd8.png)
![image](https://user-images.githubusercontent.com/44675799/145049198-fe053e5a-368e-46ed-ab68-d8200842ead2.png)
![image](https://user-images.githubusercontent.com/44675799/145049139-fd24efed-db27-462c-8ee5-20a9189d616c.png)
## Race
![image](https://user-images.githubusercontent.com/44675799/145693626-6a89fe6a-f469-44e5-9d78-4b1a7b8b86a4.png)

