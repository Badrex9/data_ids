import numpy as np

X_input = np.load('./X_input_split_train_n1/X_input_0.npy')
d_model = np.shape(X_input)[1]
for i in range(1, 20):
    X_input = np.concatenate((X_input, np.load('./X_input_split_train_n1/X_input_'+str(i)+'.npy')))
Y = np.load('./X_input_split_train/Y.npy')

X_input_new = []
Y_new = []

i=0
for raw in X_input:
    if (Y[i]==12 or Y[i]==13 or Y[i]==14):
        X_input_new.append(raw)
        Y_new.append(Y[i])
    i = i +1

np.save('./X_input_little/X_input_new.npy', X_input_new)
np.save('./X_input_little/Y_new.npy', Y_new)