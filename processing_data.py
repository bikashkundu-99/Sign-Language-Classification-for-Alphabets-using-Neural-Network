#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Load the Land handmark model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
# read image from directory
data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
#converting image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Process image and detect land handmark 
        results = hands.process(img_rgb)
# Storing the hand landmarks(x and y) of the images in an array. This array will represent the image. And the labels will be the name of the directory the image is present in:
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)
# Saving the data in filename data.pickle and creating a dictionary with keys ‘data’ and ‘labels’:
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()


# In[ ]:




