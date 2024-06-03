from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.losses import mse
import numpy as np
from tensorflow.keras.layers import Input, Dense, Lambda, Conv2D, Flatten, Reshape, Conv2DTranspose,BatchNormalization,LeakyReLU,Dropout
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

Labels = [0, 3, 4, 5 ,6, 7, 8, 9 , 10, 11, 12, 13, 14]

for label in Labels:
    # Encoder
    # Define input shape and latent dimension
    latent_dim = 2
    input_shape = (82, 20, 1)
    # Encoder network
    inputs = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    shape_before_flattening = K.int_shape(x)
    x = Flatten()(x)
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    # Sampling function
    @tf.function
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # Reparameterization trick
    z = Lambda(sampling)([z_mean, z_log_var])

    # Decoder network
    decoder_input = Input(K.int_shape(z)[1:])
    x = Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
    x = Reshape(shape_before_flattening[1:])(x)
    x = Conv2DTranspose(128, (2, 2), activation='relu', padding='same', )(x)
    x = Conv2DTranspose(64, (2, 2), activation='relu', padding='same', strides=(2, 2))(x)
    x = Conv2DTranspose(32, (2, 2), activation='relu', padding='same', )(x)
    x = Conv2DTranspose(16, (2, 2), activation='relu', padding='same', )(x)
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Define the VAE model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    decoder = Model(decoder_input, x, name='decoder')
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')

    # Define the VAE loss function
    reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss *= input_shape[0] * input_shape[1] * input_shape[2]
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)
    B = 1000   
    vae_loss = K.mean(B * reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.add_metric(kl_loss, name="kl_loss")
    vae.add_metric(reconstruction_loss, name="reconstruction_loss")
    vae.compile(optimizer='adam')

    decoder.summary()

    Y_label = label

    dataset_to_augment = np.load('./X_labels/X_label_' + str(Y_label) +'.npy')

    print(np.shape(dataset_to_augment))

    vae.fit(dataset_to_augment, epochs=500, batch_size=256)

    num_samples = 99
    random_latent_vectors  = np.random.random((num_samples, 82, 20, 1))
    random_latent_vectors = np.concatenate((np.zeros((1,82,20,1)),random_latent_vectors))

    decoded_imgs = vae.predict(random_latent_vectors)

    np.save('./generate/X_generate_' + str(Y_label) + '.npy', decoded_imgs)
    np.save('./generate/X_true_' + str(Y_label) + '.npy', dataset_to_augment[:100])
