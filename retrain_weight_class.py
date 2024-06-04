import numpy as np
from keras.models import load_model
from tensorflow.keras import datasets, layers, models
import tensorflow as tf

def train_model(X,Y, model_old, epochs=20):
    from keras import optimizers
    class_weight = {0: 1.,
                1: 1.,
                2: 1.,
                3: 1., 
                4: 1.,
                5: 1.,
                6: 1.,
                7: 1.,
                8: 1.,
                9: 1.,
                10: 1.,
                11: 1.,
                12: 1.,
                13: 30.,
                14: 1.,
                }
    model_old.compile(optimizer=optimizers.Adam(learning_rate=1e-6),  #On baisse le lr
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    #print(sys.getsizeof(model))
    model_old.fit(X, Y, epochs=epochs, batch_size=256, class_weight=class_weight)#, batch_size=65536)
    model_old.save('../Y_prediction/model_retrain_weight_classes.h5')
    #model.save('../content/drive/MyDrive/Stage sherbrooke/Model/saved_models/model_dh_30_lr_1e-5
    return model_old


print("--------------------Chargement des données train--------------------")
X_input = np.load('./X_input_split_train_n0/X_input_0.npy')
d_model = np.shape(X_input)[1]
for i in range(1, 20):
    X_input = np.concatenate((X_input, np.load('./X_input_split_train_n0/X_input_'+str(i)+'.npy')))
Y = np.load('./X_input_split_train_n0/Y.npy')
print("--------------------Fin du chargement des données--------------------")

model = load_model('./Y_prediction/model_2D_cnn_retrain_11.h5')  #Meilleur modèle

print("--------------------Entrainement du modèle--------------------")
model = train_model(X_input,Y, model, epochs=20)
