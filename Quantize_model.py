import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from sklearn.preprocessing import minmax_scale
from sklearn import preprocessing
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
import glob
import random
from sklearn.model_selection import train_test_split
from keras.models import load_model
from tensorflow import keras

def train_model(X,Y, model_old, epochs=20):
    model_old.summary()
    from keras import optimizers

    model.compile(optimizer=optimizers.Adam(learning_rate=1e-6),  #On baisse le lr
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    #print(sys.getsizeof(model))
    model.fit(X, Y, epochs=epochs, batch_size=256)#, batch_size=65536)
    model.save('./Y_prediction/model_CNN_2D_n0_avg_filter_new_data_retrain_heartbleed.h5')
    #model.save('../content/drive/MyDrive/Stage sherbrooke/Model/saved_models/model_dh_30_lr_1e-5
    return model

def split_npy_save(array, number_of_files, folder):
    file_name = 'X_input'
    i=0
    mem = 0
    len_array = len(array)
    for i in range(number_of_files):
        np.save('./'+folder+'/X_input_'+str(i)+'.npy', array[mem:int((i+1)*len_array/number_of_files)])
        mem =int((i+1)*len_array/number_of_files)

d_historique = 20

print("--------------------Chargement des données train--------------------")
X_input = np.load('./X_input_split_train_n1/X_input_0.npy')
d_model = np.shape(X_input)[1]
for i in range(1, 20):
    X_input = np.concatenate((X_input, np.load('./X_input_split_train_n1/X_input_'+str(i)+'.npy')))
Y = np.load('./X_input_split_train_n1/Y.npy')
print("--------------------Fin du chargement des données--------------------")



model = load_model('./Y_prediction/model_CNN_2D_n0_avg_filter_new_data.h5', compile=False)


import tensorflow_model_optimization as tfmot

quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

q_aware_model.summary()

q_aware_model.fit(X_input, Y,
                  batch_size=256, epochs=1)


model.save('./Y_prediction/q_aware_model.h5')

