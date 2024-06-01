import torch
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from pyraug.trainers.training_config import TrainingConfig
from pyraug.pipelines.training import TrainingPipeline

from pyraug.pipelines.generation import GenerationPipeline
from pyraug.models import RHVAE
import os

Y_label = 13

dataset_to_augment = np.load('./X_mixed_' + str(Y_label) +'.npy')

torch.cuda.is_available()

config = TrainingConfig(
    output_dir='my_model',
    train_early_stopping=50,
    learning_rate=1e-3,
    batch_size=200, # Set to 200 for demo purposes to speed up (default: 50)
    max_epochs=1000 # Set to 500 for demo purposes. Augment this in your case to access to better generative model (default: 20000)
)
torch.manual_seed(8)

# This creates the Pipeline
pipeline = TrainingPipeline(training_config=config)

# This will launch the Pipeline on the data
pipeline(train_data=dataset_to_augment, log_output_dir='output_logs')

last_training = sorted(os.listdir('my_model'))[-1]

# reload the model
model = RHVAE.load_from_folder(os.path.join('my_model', last_training, 'final_model'))

torch.manual_seed(88)
torch.cuda.manual_seed(88)

# This creates the Pipeline
generation_pipe = GenerationPipeline(
    model=model
)

# This will launch the Pipeline
len_generation = 100000
generation_pipe(len_generation)

last_generation = sorted(os.listdir('dummy_output_dir'))[-1]
generated_data = torch.load(os.path.join('dummy_output_dir', last_generation, 'generated_data_500_0.pt'))

for i in range(1,20):
    generated_data = torch.cat((generated_data, torch.load(os.path.join('dummy_output_dir', last_generation, 'generated_data_500_' + str(i) + '.pt'))), 0)
    
np.save('./X_autoencoder_' + str(Y_label) +'.npy', generated_data.view(len_generation, 82, 20).cpu().numpy())
np.save('./Y_autoencoder_' + str(Y_label) +'.npy', [Y_label for i in range(len_generation)])

import numpy as np
from keras.models import load_model
from tensorflow.keras import datasets, layers, models
import tensorflow as tf

def train_model(X,Y, model_old, epochs=20):
    from keras import optimizers
    model_old.compile(optimizer=optimizers.Adam(learning_rate=1e-6),  #On baisse le lr
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    #print(sys.getsizeof(model))
    model_old.fit(X, Y, epochs=epochs, batch_size=256)#, batch_size=65536)
    model_old.save('../Y_prediction/model_autoencoder_2_100000.h5')
    #model.save('../content/drive/MyDrive/Stage sherbrooke/Model/saved_models/model_dh_30_lr_1e-5
    return model_old


print("--------------------Chargement des données train--------------------")
X_input = np.load('../X_input_split_train_n0/X_input_0.npy')
d_model = np.shape(X_input)[1]
for i in range(1, 20):
    X_input = np.concatenate((X_input, np.load('../X_input_split_train_n0/X_input_'+str(i)+'.npy')))
Y = np.load('../X_input_split_train_n0/Y.npy')
print("--------------------Fin du chargement des données--------------------")


X_autoencoder = np.load('./X_autoencoder_' + str(Y_label) +'.npy')
Y_autoencoder = np.load('./Y_autoencoder_' + str(Y_label) +'.npy')

#Concat
X_input = np.concatenate((X_input, X_autoencoder))
Y = np.concatenate((Y, Y_autoencoder))

model = load_model('../Y_prediction/model_2D_cnn_retrain_11.h5')  #Meilleur modèle

print("--------------------Entrainement du modèle--------------------")
model = train_model(X_input,Y, model, epochs=20)

d_historique = 20

print("--------------------Load test--------------------")
X_input_test = np.load('../X_input_split_test_n0/X_input_0.npy')
for i in range(1, 10):
    X_input_test = np.concatenate((X_input_test, np.load('../X_input_split_test_n0/X_input_'+str(i)+'.npy')))
Y_test = np.load('../X_input_split_test_n0/Y_test.npy')

d_model = np.shape(X_input_test)[1]
X_input_test = X_input_test.reshape(np.shape(X_input_test)[0],d_model,d_historique,1)

print("--------------------Prédiction du modèle--------------------")
prediction = model.predict(X_input_test)

y_prediction = prediction.argmax(axis=1)
np.save('./Y_prediction_autoencoder_2_100000.npy', y_prediction)
