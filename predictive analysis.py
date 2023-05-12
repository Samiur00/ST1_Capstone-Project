#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def preprocess_image(img_path):
    # Load the image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Resize the image to a standard size
    img = cv2.resize(img, (224, 224))
    # Flatten the image into a 1D array
    img = img.flatten()
    # Normalize the pixel values to be between 0 and 1
    img = img.astype('float32') / 255.0
    return img


import os

# Define the image directories
lion_dir = 'C:/Users/ssann/Desktop/University/Software Technology 1/images/Lions'
cheetah_dir = 'C:/Users/ssann/Desktop/University/Software Technology 1/images/Cheetahs'

# Load the images and preprocess them
X = []
y = []
for filename in os.listdir(lion_dir):
    img_path = os.path.join(lion_dir, filename)
    X.append(preprocess_image(img_path))
    y.append(0)  # 0 for lion
for filename in os.listdir(cheetah_dir):
    img_path = os.path.join(cheetah_dir, filename)
    X.append(preprocess_image(img_path))
    y.append(1)  # 1 for cheetah
# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train a logistic regression model
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

# Evaluate the model on the testing data
accuracy = lr.score(X_test, y_test)
print('Accuracy:', accuracy)


y_pred = lr.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix:')
print(cm)


# In[ ]:




