# Machine learning Image Classification project using face and eye detection.
In this machine learning project, with the help of OpenCV Face and eye detection, we classify world leader personalities. We restrict classification to only 5 people. Create a web page and real-time  frontend image classification and backend response from probability output from machine learning.

Here is the folder structure:
  google_image_scrapping: code to scrap Google for images
  images_dataset: Dataset used for our model training
  model: Contains Python notebook for model building
  UI : This contains ui website code
  server: Python flask server

Technologies used in this project,

Python
Numpy and OpenCV for data cleaning
Matplotlib & Seaborn for data visualization
Sklearn for model building
Jupyter notebook, visual studio code and pycharm as IDE
Python flask for http server
HTML/CSS/Javascript for UI

Step 1: Collect Images from Google and prepare the dataset.

Step 2: Face and Eye detection using OpenCV. When we look at any image, most of the time we identify a person using a face. An image might contain multiple faces, also the face can be obstructed and not clear. The first step in our pre-processing pipeline is to detect faces from an image. Once face is detected, we will detect eyes, if two eyes are detected then only we keep that image otherwise discard it.

![Face and eye detection](https://github.com/abulzunayed/Machine_learning_Projects/assets/122612945/963054e9-9a88-48d6-8d50-c26275967c21)

Step 3: Create machine learning models using GridSearch and calculate accuracy measurements.

Final Step: Create a web page and real-time  frontend image classification and backend response from probability output from machine learning.

![Web_front_page](https://github.com/abulzunayed/Machine_learning_Projects/assets/122612945/dde676d7-75b7-4cca-92f1-47c8fae4f63f)


