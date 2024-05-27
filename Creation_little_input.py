import numpy as np

X_input = np.load('./X_input_split_train_n0/X_input_0.npy')
d_model = np.shape(X_input)[1]
for i in range(1, 20):
    X_input = np.concatenate((X_input, np.load('./X_input_split_train_n0/X_input_'+str(i)+'.npy')))
Y = np.load('./X_input_split_train/Y.npy')

nombre_donnees = 1000

X_input_new = []
Y_new = []

for k in tqdm(range(15)):
    i =0
    j = 0
    while (i<nombre_donnees and j<np.shape(X_input)[0]):
        if (Y[j]==k):
            X_input_new.append(X_input[j])
            Y_new.append(Y[j])
            i = i +1
        j = j+1


np.save('./X_input_little/X_input_new.npy', X_input_new)
np.save('./X_input_little/Y_new.npy', Y)