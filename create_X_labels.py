import numpy as np

print("--------------------Chargement des données train--------------------")
X_input = np.load('../X_input_split_train_n0/X_input_0.npy')
d_model = np.shape(X_input)[1]
for i in range(1, 20):
    X_input = np.concatenate((X_input, np.load('../X_input_split_train_n0/X_input_'+str(i)+'.npy')))
Y = np.load('../X_input_split_train_n0/Y.npy')
print("--------------------Fin du chargement des données--------------------")

Labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

longueur = np.shape(X_input)[0]

for label in Labels:
    X_new = []
    for i in range(longueur):
        if Y[i]==label:
            X_new.append(X_input[i])
    np.save('./X_labels/X_label_' + str(label) + '.npy', X_new)
