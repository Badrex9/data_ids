import numpy as np
from keras.models import load_model
from tensorflow.keras import datasets, layers, models
import tensorflow as tf

d_historique = 20

print("--------------------Load test--------------------")
X_input_test = np.load('./X_input_split_test_n0/X_input_0.npy')
for i in range(1, 10):
    X_input_test = np.concatenate((X_input_test, np.load('./X_input_split_test_n0/X_input_'+str(i)+'.npy')))
Y_test = np.load('./X_input_split_test_n0/Y_test.npy')


d_model = np.shape(X_input_test)[1]
X_input_test = X_input_test.reshape(np.shape(X_input_test)[0],d_model,d_historique,1)


model = load_model('./Y_prediction/model_retrain_weight_classes.h5')

print("--------------------Prédiction du modèle--------------------")
prediction = model.predict(X_input_test)

y_prediction = prediction.argmax(axis=1)
np.save('./Y_prediction/Y_pred_class_weight.npy', y_prediction)

