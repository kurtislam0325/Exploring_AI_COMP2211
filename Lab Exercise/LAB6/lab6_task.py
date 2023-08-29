# You can import more modules to use in a later stage here
# But REMEMBER to add the import line(s) also in the .py file before submission
import pandas as pd
import numpy as np
import re
import keras
import os

from keras.preprocessing.text import one_hot
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, GRU, Bidirectional, LSTM
from keras.layers.core import Activation, Dropout, Dense
from keras.initializers import HeNormal
from keras.regularizers import l1, l2

import nltk
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


def process_text(text):
    # Input: text string, a Pandas DataFrame's column
    # Return: processed text string
    ps = PorterStemmer()
    lemmatizer=WordNetLemmatizer()
    names = ['delta', 'deltaair', 'united', 'unitedair', 'southwest', 'southwestair', 'usairways',
         'virginamerica', 'american', 'americanair', 'jetblue', 'jetblues', 'usairway',
         'flight', 'airline', 'airlines']
    stopword = stopwords.words('english')
    for name in names:
        stopword.append(name)
    review = re.sub(r'[^a-zA-Z\s]','',text)
    review = review.lower()
    review= review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in stopword]
    review = ' '.join(review)
    return review


def tokenized(text, max_length, vocab_size):
    # Input: 
    # text: text strings, can be a Python list or a Pandas DataFrame's column
    # max_length: the max lenth of a text string, int
    # vocab_size: the vocabulary size, int
    # Return: the tokenized text, a Numpy 2D array, shape: (len(text), max_length)
    onehot_repr=[one_hot(words,vocab_size)for words in text] 
    onehot_padded = pad_sequences(onehot_repr, padding='post', maxlen=max_length)
    return onehot_padded


def Preprocess(data, max_length, vocab_size, test_rate):
  # Step 1: Fill NA/NaN values using '' in the column data['negativereason'].
  data['negativereason'] = data['negativereason'].fillna('')

  # TODO
  # Step 2: Process the text data in data['processed_text'] and data['processed_text'].
  # Hint 1: data['processed_text'] is processed from data['text']. data['negativereason'] is processed from data['negativereason'].
  # Hint 2: Please use the function process_text(text) in the above code cell to process the text data.
  # Hint 3: You may use Pandas's DataFrame.apply() function.
  # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html
  data['processed_text'] = data['text'].apply(process_text)
  data['negativereason'] = data['negativereason'].apply(process_text)

  # TODO
  # Step 3: Create a new column for data, where each entry of data['final_text'] is produced by concatenation of the corresponding contents in data['processed_text'] and data['negativereason'], separated with one blank ' '.
  data['final_text'] = data['processed_text'] + " " + data['negativereason']

  # TODO
  # Step 4: Tokenize data['final_text']. X should be a Numpy 2D array with shape (num_data, max_length).
  # Hint: Please use the function tokenized(text, max_length, vocab_size) in the above cell to tokenize the final text data.
  X = tokenized(data['final_text'], max_length , vocab_size)
  
  # TODO
  # Step 5: Get a new DataFrame that records the labels for the data samples. 
  # y should be a Pandas DataFrame with shape (num_data, 3). y has 3 columns. Each column represents a class label (negative, neutral, positive).
  # In each row (i.e., each data sample), only one entry has value 1, while the other 2 entries are 0, indicating the data sample is labeled as the corresponding column.
  # Hint 1: Please get y from data['airline_sentiment'].
  # Hint 2: You may use Pandas's get_dummies() function.
  # https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
  y = pd.get_dummies(data['airline_sentiment'], dtype = int)

  # Step 6: Split X and y to training and testing datasets.
  X_train,X_test,y_train, y_test = train_test_split(X, y, test_size=test_rate, random_state=42)
  return X_train, X_test, y_train, y_test


def myModel(vocab_size, maxlen, embed_dim):
  model = Sequential()
  embedding_layer = Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length = maxlen)
  model.add(embedding_layer)
  # TODO: Please construct your MLP model. You can try different settings on your own.
  # Remark: Your model will be trained with 10 epochs under the same training setting as the notebook (e.g. the same training set, training epochs, optimizer etc.) for evaluation.
  output_layers_neurons = 3
  num_neurons = int(np.sqrt(vocab_size * output_layers_neurons)) + 2
  
  forward_layer = LSTM(units = num_neurons, return_sequences=True)
  backward_layer = LSTM(units = num_neurons, activation='relu', return_sequences=True,
                      go_backwards=True)
  initializers = keras.initializers.HeNormal()

  model.add(Bidirectional(forward_layer, backward_layer = backward_layer, input_shape=(num_neurons,)))
  model.add(Flatten())
  model.add(Dense(units = num_neurons, activation = "relu", 
            kernel_initializer = initializers,
            kernel_regularizer = keras.regularizers.l2(0.01),
            activity_regularizer = keras.regularizers.l1(0.01))) 
  model.add(Dropout(.2, input_shape=(num_neurons,)))

  model.add(Dense(units = output_layers_neurons, activation = 'softmax'))
  return model

  return model