

import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import keras 
import keras.backend as K
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.layers import Flatten, Reshape
from tensorflow.keras.layers import BatchNormalization, Dropout

def dropout_and_batchnorm(x):
    return Dropout(0.3)(BatchNormalization()(x))

def noiser(args):
    global mean, log_var
    mean, log_var = args
    N = K.random_normal(shape=(batch_size, hidden_dim), mean=0., stddev=1.0)
    return K.exp(log_var / 2) * N + mean


hidden_dim = 2
batch_size = 128


label = 12

input_shape = (82, 20, 1)
input_image = Input(shape=input_shape)
x = Flatten()(input_image)
x = Dense(256, activation="relu")(x)
x = dropout_and_batchnorm(x)
x = Dense(128, activation="relu")(x)
x = dropout_and_batchnorm(x)

mean = Dense(hidden_dim)(x)
log_var = Dense(hidden_dim)(x)
h = Lambda(noiser, output_shape=(hidden_dim), name="latent_space")([mean, log_var])



input_decoder = Input(shape=(hidden_dim,))
d = Dense(128, activation="relu")(input_decoder)
d = dropout_and_batchnorm(d)
d = Dense(256, activation="relu")(d)
d = dropout_and_batchnorm(d)
d = Dense(82*20, activation="sigmoid")(d)
decoded = Reshape((82, 20, 1))(d)

def vae_loss(x, y):
    x = K.reshape(x, shape=(batch_size, 82*20))
    y = K.reshape(y, shape=(batch_size, 82*20))
    loss = K.sum(K.square(x-y), axis=-1)
    kl_loss = -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=-1)
    return loss + kl_loss


encoder = keras.Model(input_image, h, name="encoder")
decoder = keras.Model(input_decoder, decoded, name="decoder")
vae = keras.Model(input_image, decoder(encoder(input_image)), name="vae")
vae.compile(optimizer="adam", loss=vae_loss)
vae.summary()


dataset_to_augment = np.load('./X_labels/X_label_' + str(label) +'.npy')
input_dimension = np.shape(dataset_to_augment)
print(input_dimension[0])
dataset_to_augment.reshape((input_dimension[0],input_dimension[1],input_dimension[2],1))
reste = np.shape(dataset_to_augment)[0] - np.shape(dataset_to_augment)[0]%batch_size
dataset_to_augment = dataset_to_augment[:reste]


vae.fit(dataset_to_augment,dataset_to_augment,
        epochs=1000,
        batch_size=batch_size,
        shuffle=True
    )

num_samples = 8191
random_latent_vectors  = np.random.random((num_samples, 82, 20, 1))
random_latent_vectors = np.concatenate((np.zeros((1,82,20,1)),random_latent_vectors))
decoded_imgs = vae.predict(random_latent_vectors, batch_size=batch_size)
print(decoded_imgs)
np.save('./generate/X_generate_' + str(label) + '.npy', decoded_imgs)



