import random
import numpy as np
import tqdm

X_input = np.load('./X_input_new.npy')
Y = np.load('./Y_new.npy')

X_new=[]
Y_new=[]
len_x = np.shape(X_input)[0]
i=0
for i in tqdm(range(len_x)):
    indice = random.randint(0, len_x-i-1)
    X_new.append(X_input[i])
    Y_new.append(Y[i])
    X_input = np.delete(X_input, i,0)
    Y = np.delete(Y, i,0)
    i = i+1

np.save("./X_mixed.npy", X_input)
np.save("./Y_mixed.npy", Y_new)

print(Y_new)