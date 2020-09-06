
# coding: utf-8

# In[31]:


import numpy as np
import os
import cv2
import keras
import tensorflow.keras
from tensorflow.keras.utils import Sequence
import math


class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, img_list, label, batch_size):
        self.img_list, self.label = img_list, label
        self.batch_size = batch_size

    def __len__(self):
        #return np.ceil(len(self.img_list) / float(self.batch_size))
        return math.ceil(len(self.img_list) / float(self.batch_size))

    def __getitem__(self, idx):
        
        img_list = self.img_list
        label = self.label
        batch_size = self.batch_size
        
        imgs = list()
        labels = list()
        input_len = list()
        label_len = list()
        
        for i in range(idx * self.batch_size,(idx + 1) * self.batch_size):
            if i >= len(img_list):
                break
                
            temp = cv2.imread(img_list[i], cv2.IMREAD_GRAYSCALE)
            temp = temp.reshape(temp.shape + (1,))
            temp = temp / 255.0
            imgs.append(temp)
            labels.append(label[i])
            input_len.append(31)
            label_len.append(len(label[i]))


        data={
            'image_input': np.array(imgs),
            'label_input': np.array(labels),
            'input_length': np.array(input_len), # used by ctc
            'label_length': np.array(label_len), # used by ctc
            
#            'source_str': source_str, # used for visualization only
        }
        
        outputs_dict = {'ctc': np.zeros([batch_size])}  # dummy
        
        return data, outputs_dict
        #yield data, outputs_dict

