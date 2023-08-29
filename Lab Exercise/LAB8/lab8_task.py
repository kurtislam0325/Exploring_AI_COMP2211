# -*- coding: utf-8 -*-
"""
Submission Template for Lab 8
Important notice: DO NOT use any global variables in this submission file
"""

# Task 1
def data_preprocessing(data_dir, cate2Idx, img_size):
  x = []
  y = []
  ###############################################################################
  # TODO: your code starts here

  # TODO: your code ends here
  ###############################################################################
  x = np.asarray(x)
  y = np.asarray(y)
  return x, y


# Task 2
def get_datagen():
  datagen = None
  ###############################################################################
  # TODO: your code starts here

  # TODO: your code ends here
  ###############################################################################
  return datagen


# Task 3
def custom_model():
  model = None
  ###############################################################################
  # TODO: your code starts here

  # TODO: your code ends here
  ###############################################################################
  return model


if __name__ == '__main__':
  # Import necessary libraries
  import os, cv2
  import numpy as np
  from sklearn.model_selection import train_test_split
  import keras
  from keras.utils import np_utils
  from keras.models import Sequential
  from keras.layers import Conv2D, MaxPooling2D
  from keras.layers import Dense, Dropout, Flatten
  from keras.preprocessing.image import ImageDataGenerator
