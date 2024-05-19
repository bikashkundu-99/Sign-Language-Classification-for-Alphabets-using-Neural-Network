#!/usr/bin/env python
# coding: utf-8

# In[3]:

# function for computing accuracy
def compute_accuracy(Y_true, Y_pred):  
    correctly_predicted = 0  
    # iterating over every label and checking it with the true sample  
    for true_label, predicted in zip(Y_true, Y_pred):  
        if true_label == predicted:  
            correctly_predicted += 1  
    # computing the accuracy score  
    accuracy_score = correctly_predicted / len(Y_true)  
    return accuracy_score  


# In[4]:


import pickle
from sklearnex import patch_sklearn
patch_sklearn()


from sklearnex.ensemble import RandomForestClassifier
from sklearnex.model_selection import train_test_split
from sklearnex import metrics
import numpy as np

# Loading the data and converting the ‘data’ and ‘label’ list into numpy arrays:
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
# Splitting the datasets into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Creating the model using random forest classifier and training the model with the training dataset
model = RandomForestClassifier()

model.fit(x_train, y_train)
# Making predictions on new data points
y_predict = model.predict(x_test)
# Computing the accuracy of the model
score = compute_accuracy(y_test,y_predict)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()


# In[ ]:




