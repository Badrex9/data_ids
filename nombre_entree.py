import numpy as np

Y_label = 13

print("--------------------Chargement des données train--------------------")
X_input = np.load('../X_input_split_train_n0/X_input_0.npy')
d_model = np.shape(X_input)[1]
for i in range(1, 20):
    X_input = np.concatenate((X_input, np.load('../X_input_split_train_n0/X_input_'+str(i)+'.npy')))
Y = np.load('../X_input_split_train_n0/Y.npy')
print("--------------------Fin du chargement des données--------------------")


X_autoencoder = np.load('./X_autoencoder_' + str(Y_label) +'.npy')
Y_autoencoder = np.load('./Y_autoencoder_' + str(Y_label) +'.npy')

print("X_input1 :", np.shape(X_input))

X_input = np.concatenate((X_input, X_autoencoder))
Y = np.concatenate((Y, Y_autoencoder))

print("X_input concat :", np.shape(X_input))