import cv2
import numpy as np

def appliquer_filtre_1xn(matrice, taille_filtre):
    # Créer un noyau 1xn rempli de 1s
    filtre = np.ones((1, taille_filtre), dtype=np.float64) / taille_filtre

    # Appliquer le filtre à la matrice
    image_filtree = cv2.filter2D(matrice, -1, filtre)
    return image_filtree

def split_npy_save(array, number_of_files, folder):
    file_name = 'X_input'
    i=0
    mem = 0
    len_array = len(array)
    for i in range(number_of_files):
        np.save('./'+folder+'/X_input_'+str(i)+'.npy', array[mem:int((i+1)*len_array/number_of_files)])
        mem =int((i+1)*len_array/number_of_files)

print("--------------------Chargement des données train--------------------")
X_input = np.load('./X_input_split_train_n0/X_input_0.npy')
d_model = np.shape(X_input)[1]
for i in range(1, 20):
    X_input = np.concatenate((X_input, np.load('./X_input_split_train_n0/X_input_'+str(i)+'.npy')))
print("--------------------Fin du chargement des données--------------------")


d_model = np.shape(X_input)[1]
for mat in X_input:
    mat = appliquer_filtre_1xn(mat, d_model)
    
#Save data
split_npy_save(X_input, 20, 'X_input_split_train_filter_n0')

print("--------------------Load test--------------------")
X_input_test = np.load('./X_input_split_test_n0/X_input_0.npy')
for i in range(1, 10):
    X_input_test = np.concatenate((X_input_test, np.load('./X_input_split_test_n0/X_input_'+str(i)+'.npy')))

for mat in X_input_test:
    mat = appliquer_filtre_1xn(mat, d_model)

split_npy_save(X_input_test, 20, 'X_input_split_test_filter_n0')
