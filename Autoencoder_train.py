import numpy as np
from keras.models import load_model
from tensorflow.keras import datasets, layers, models
import tensorflow as tf

Y_label = 13

def train_model(X,Y, model_old, epochs=20):
    from keras import optimizers
    model_old.compile(optimizer=optimizers.Adam(learning_rate=1e-6),  #On baisse le lr
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    #print(sys.getsizeof(model))
    model_old.fit(X, Y, epochs=epochs, batch_size=256)#, batch_size=65536)
    model_old.save('./Y_prediction/model_autoencoder_1.h5')
    #model.save('../content/drive/MyDrive/Stage sherbrooke/Model/saved_models/model_dh_30_lr_1e-5
    return model_old


print("--------------------Chargement des données train--------------------")
X_input = np.load('./X_input_split_train_n0/X_input_0.npy')
d_model = np.shape(X_input)[1]
for i in range(1, 20):
    X_input = np.concatenate((X_input, np.load('./X_input_split_train_n0/X_input_'+str(i)+'.npy')))
Y = np.load('./X_input_split_train_n0/Y.npy')
print("--------------------Fin du chargement des données--------------------")


X_autoencoder = np.load('./X_input_split_train_n0/X_autoencoder_' + str(Y_label) +'.npy')
Y_autoencoder = np.load('./X_input_split_train_n0/Y_autoencoder_' + str(Y_label) +'.npy')

#Concat
X_input = np.concatenate((X_input, X_autoencoder))
Y = np.concatenate((Y, Y_autoencoder))

model = load_model('./Y_prediction/model_2D_cnn_retrain_11.h5')  #Meilleur modèle

print("--------------------Entrainement du modèle--------------------")
model = train_model(X_input,Y, model, epochs=20)