# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 11:50:28 2024

@author: Zunayed
"""

import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
#%matplotlib inline
print(" Test your jupyter file ")

img = cv2.imread("E:/World_leader_classification/model/test_images/Emmanuel_Macron.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   ## make it into color image
img.shape  # get shape of image

plt.imshow(img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray.shape

gray

plt.imshow(gray, cmap='gray')

face_cascade = cv2.CascadeClassifier('E:/World_leader_classification/model/opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('E:/World_leader_classification/model/opencv/haarcascades/haarcascade_eye.xml')

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
faces  # this will give 4D value,-> Out[16]: array([[352,  38, 233, 233]], dtype=int32)indicate { x, y, width, heght}

# Save the value axis wise
(x,y,w,h) = faces[0]
x,y,w,h

# Draw an red color Box on Face of image
face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
plt.imshow(face_img)

## Draw Box on eye image
cv2.destroyAllWindows()
for (x,y,w,h) in faces:
    face_img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = face_img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        

plt.figure()
plt.imshow(face_img, cmap='gray')
plt.show()

#%matplotlib inline
plt.imshow(roi_color, cmap='gray')

# get eye shape size
cropped_img = np.array(roi_color)
cropped_img.shape

import numpy as np
import pywt
import cv2    

def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255;
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H

im_har = w2d(cropped_img,'db1',5)
plt.imshow(im_har, cmap='gray')



### defined function for crop Face

def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color
        
        
## original image
original_image = cv2.imread("E:/World_leader_classification/model/test_images/Emmanuel_Macron.jpg")
plt.imshow(original_image)

## Get only crop image
cropped_image = get_cropped_image_if_2_eyes("E:/World_leader_classification/model/test_images/Emmanuel_Macron.jpg")
plt.imshow(cropped_image)

## original image
org_image_obstructed = cv2.imread("E:/World_leader_classification/model/test_images/Sheikh_Hasina.jpg")
plt.imshow(org_image_obstructed)

###   save all crop eye from in´mages in the sub folder as per serial/ label wise
## create crop folder in current directory folder
path_to_data = "E:/World_leader_classification/model/dataset/"
path_to_cr_data = "E:/World_leader_classification/model/cropped/"


## create python list via os , so that, can create sub folder in the mention path
import os
img_dirs = []
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)
        
# print the all subdirectory vaiables
img_dirs

# create sub crop folder in current dataset folder
import shutil
if os.path.exists(path_to_cr_data):
     shutil.rmtree(path_to_cr_data)
os.mkdir(path_to_cr_data)           ## os.makedir() create folder


## iterarte all image directory and create folder name as per celebraty, finally crop eye from all images.
cropped_image_dirs = []
celebrity_file_names_dict = {}
for img_dir in img_dirs:
    count = 1
    celebrity_name = img_dir.split('/')[-1]

     # upto here just create subfolder name.
    # Now below loop will give you crop eye image in those subfolder
    celebrity_file_names_dict[celebrity_name] = []
    for entry in os.scandir(img_dir):
       # print(entry.path) ### check name of each image
        roi_color = get_cropped_image_if_2_eyes(entry.path)       # get crop face image as label/ serial wise
        if roi_color is not None:
            cropped_folder = path_to_cr_data + celebrity_name
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                cropped_image_dirs.append(cropped_folder)
                print("Generating cropped images in folder: ",cropped_folder)
            cropped_file_name = celebrity_name + str(count) + ".png"
            cropped_file_path = cropped_folder + "/" + cropped_file_name
            cv2.imwrite(cropped_file_path, roi_color)             # write image
            celebrity_file_names_dict[celebrity_name].append(cropped_file_path)  ### create dictionary for label images
            count += 1
            
            
### This dictinany give us all crop image
celebrity_file_names_dict = {}
###  This dictionary will use in next model, It contains model name wise dicrionary and List of all crop image.###

for img_dir in cropped_image_dirs:
    celebrity_name = img_dir.split('/')[-1]
    file_list = []
    for entry in os.scandir(img_dir):
        file_list.append(entry.path)
    celebrity_file_names_dict[celebrity_name] = file_list
celebrity_file_names_dict

### Assiagn celebratiy name into number wise
class_dict = {}
count = 0
for celebrity_name in celebrity_file_names_dict.keys():
    class_dict[celebrity_name] = count
    count = count + 1
class_dict



X, y = [], []
for celebrity_name, training_files in celebrity_file_names_dict.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img,'db1',5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
        X.append(combined_img)
        y.append(class_dict[celebrity_name]) 


# watch out length of List X or number of total List
len(X[0])

# Size of each image of each list of image
len(X[0])   #  output: (raw_image 32*32*3)+(wavelet_imag 32*32)

## watch out each image of each list of X
X[0]

## watch out each image of each list of Y
y[0]

# reshape and convert into float for assurance the correct shape (182, 4096) with float number
X = np.array(X).reshape(len(X),4096).astype(float)
X.shape

## import all module for createing a model
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


##  Splite train and test date
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)


# check accuracy without NN model
print(classification_report(y_test, pipe.predict(X_test)))  ## check accuracy 82% without  Nueral network

### Use GridSearchCV to get almost good hyperparameter for model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# create 3 different models for SVM, Random Forest, Logistric Regression
model_params = {
    'svm': {                                                  ## use SVM model
        'model': svm.SVC(gamma='auto',probability=True),
        'params' : {
            'svc__C': [1,10,100,1000],
            'svc__kernel': ['rbf','linear']
        }  
    },
    'random_forest': {                                          ## Use Random forest model
        'model': RandomForestClassifier(),
        'params' : {
            'randomforestclassifier__n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {                                   ## Use Logistic regression
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'logisticregression__C': [1,5,10]
        }
    }
}

# check accuracy score for 3 different models and apend all score and save
scores = []
best_estimators = {}
import pandas as pd
## iterate dictionary each of models
for algo, mp in model_params.items():
    pipe = make_pipeline(StandardScaler(), mp['model'])            # rescale all date for each of 3 models
    clf =  GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)   # Cv -> cross validation for test and train data set
    clf.fit(X_train, y_train)
    scores.append({                                           ## apend all scores
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimators[algo] = clf.best_estimator_
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])

 ## print all score
df

##   print best estimator model ****
best_estimators

## print best estimator score for SVM
best_estimators['svm'].score(X_test,y_test)

## print best estimator score for Random forest
best_estimators['random_forest'].score(X_test,y_test)

## print best estimator score for Logistic regression
best_estimators['logistic_regression'].score(X_test,y_test)

best_clf = best_estimators['svm']

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, best_clf.predict(X_test))

 # print confusing matrix
cm

# plot confusing matrix via seaborn
import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

# compare actualliy with confusion matrix and  below didctionary
class_dict

# need to install this pip for joblib->>   pip install joblib
#!pip install joblib
import joblib 
# Save the model as a pickle in a file 
joblib.dump(best_clf, 'saved_model.pkl') 

### ***********************************Save class dictionary
import json
with open("class_dictionary.json","w") as f:
    f.write(json.dumps(class_dict))


