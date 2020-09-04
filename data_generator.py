
# coding: utf-8

# In[2]:


import numpy as np
import os
import cv2
import keras


# In[3]:


class DataGenerator(keras.utils.Sequence):

    def __init__(self, img_list, label_text, batch_size):
        
        self.__dict__.update(locals())
    
    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        image = [self.img_list[k] for k in indexes]
        label = [self.label_text[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(img, label)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, image, label):
        # Initialization
        X = []
        y = []
        
        for i in range(len(image)):
            X.append(cv2.imread(image[i], cv2.IMREAD_GRAYSCALE))
            Y.append(label[i])
#         X = np.empty((self.batch_size, (32,128,1), 1))
#         y = np.empty((self.batch_size), dtype=int)

#         # Generate data
#         for i, ID in enumerate(list_IDs_temp):
#             # Store sample
#             X[i,] = np.load('data/' + ID + '.npy')

#             # Store class
#             y[i] = self.labels[ID]

#         for i, ID in enumerate(list_IDs_temp):
#             # Store sample
#             X[i,] = np.load('data/' + ID + '.npy')

#             # Store class
#             y[i] = self.labels[ID]

            
        return X, Y

