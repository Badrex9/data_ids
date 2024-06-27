import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from sklearn.preprocessing import minmax_scale
from sklearn import preprocessing
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
import glob
import random
from sklearn.model_selection import train_test_split

def importation_csv():
    # Get a list of all CSV files in a directory
    csv_files = glob.glob('./TrafficLabelling/*.csv')

    # Create an empty dataframe to store the combined data
    combined_df = pd.DataFrame()

    # Loop through each CSV file and append its contents to the combined dataframe
    for csv_file in csv_files:
        print(csv_file.title)
        df = pd.read_csv(csv_file, encoding='cp1252')
        combined_df = pd.concat([combined_df, df])
    data = combined_df

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data

def creation_X_Y_ip(data):
    # On retire la première ligne (headers) et les colonnes donc on ne se sert pas ( 'Flow ID' ' Source IP' ' Source Port' ' Destination IP' ' Destination Port' ' Protocol' ' Timestamp' 'Label')
    # On mettra séparement les colonnes port source et dest et on ajoutera le protocol en onehot encoding
    data_without_nan = data.values[~pd.isna(data.values).any(axis=1)]
    source_port = np.vstack(data_without_nan[1:,2])
    dest_port = np.vstack(data_without_nan[1:,4])

    source_ip = np.vstack(data_without_nan[1:,1])
    dest_ip = np.vstack(data_without_nan[1:,3])
    X = data_without_nan[1:,7:]
    #Conversion One Hot Encoding de la colonne Protocol
    ohe = data_without_nan[1:,5]
    #print(np.shape(ohe))
    #print(ohe)
    #ohe = ohe.replace(np.nan, 0)
    #print(np.shape(ohe))
    ohe = pd.get_dummies(ohe.astype(int), dtype=int)
    Y = np.vstack(X[:,-1])
    Y = np.array(Y)
    X = X[:,:np.shape(X)[1]-1]
    #On ajoute ces colonnes aux précédentes
    X = np.concatenate((source_port, X), axis=1)
    X = np.concatenate((dest_port, X), axis=1)
    X = minmax_scale(X, axis=0)
    X = np.concatenate((ohe.values, X), axis=1)
    return X, Y, source_ip, dest_ip, source_port, dest_port


def choix_donnees_entrainement_70_30(X, Y, source_ip, dest_ip, source_port, dest_port):
    label_encoder = preprocessing.LabelEncoder()
    Y= label_encoder.fit_transform(Y)
    source_ip = label_encoder.fit_transform(source_ip)
    dest_ip = label_encoder.fit_transform(dest_ip)
    X_train, X_test, Y_train, Y_test, source_ip_train, source_ip_test, dest_ip_train, dest_ip_test, source_port_train, source_port_test, dest_port_train, dest_port_test = train_test_split(X,Y,source_ip,dest_ip, source_port, dest_port,random_state=843,test_size=0.3, stratify=Y)


    return X_train, X_test, np.array(Y_train), np.array(Y_test), source_ip_train, source_ip_test, dest_ip_train, dest_ip_test, source_port_train, source_port_test, dest_port_train, dest_port_test

def transfo(X, source_ip, dest_ip, source_port, dest_port):
    
    src_dest = []

    flows= []

    data_input = []

    i=0
    for raw in tqdm(X):
        
        passage = False
        sip = source_ip[i]
        dip = dest_ip[i]
        sp = source_port[i]
        dp = dest_port[i]
        j=0
        for src_dest_raw in src_dest:
            
            
            sip_raw = src_dest_raw[0]
            dip_raw = src_dest_raw[1]
            sp_raw = src_dest_raw[2]
            dp_raw = src_dest_raw[3]
            
            
            if ((sp==sp_raw and dp==dp_raw) or (sp==dp_raw and dp==sp_raw)) and ((sip==sip_raw and dip==dip_raw) or (sip==dip_raw and dip==sip_raw)):
                passage = True
                new_mat = np.insert(flows[j],0, raw, axis=1)[:, :20]
                #print(new_mat)
                flows[j] = new_mat  #décalage
                data_input.append(new_mat) #Ajout à l'input
                break
            
            j+=1
        if not passage:
                new_mat = np.tile(raw, (20, 1)).T  #Créer une matrice avec 20 fois le même vecteur
                flows.append(new_mat)
                data_input.append(new_mat)
                src_dest.append([sip, dip, sp, dp])
        i+=1
    
    return(data_input)
def split_npy_save(array, number_of_files, folder):
    file_name = 'X_input'
    i=0
    mem = 0
    len_array = len(array)
    for i in range(number_of_files):
        np.save('./'+folder+'/X_input_'+str(i)+'.npy', array[mem:int((i+1)*len_array/number_of_files)])
        mem =int((i+1)*len_array/number_of_files)

print("--------------------Importation données--------------------")
data_frame = importation_csv()
print("--------------------Séparation des données--------------------")
X_data, Y_data, source_ip_data, dest_ip_data, source_port_data, dest_port_data = creation_X_Y_ip(data_frame)


#Choix des données pour l'entrainement du modèle
print("--------------------Sélection des données d'entrainement--------------------")
X, X_test, Y, Y_test, source_ip, source_ip_test, dest_ip, dest_ip_test, source_port, source_port_test, dest_port, dest_port_test = choix_donnees_entrainement_70_30(X_data, Y_data, source_ip_data, dest_ip_data, source_port_data, dest_port_data)
print("--------------------Création des tableaux 2D pour les données entrainement--------------------")


d_model = np.shape(X)[1]
d_historique = 20
X_input = transfo(X, source_ip.astype(int), dest_ip.astype(int), source_port.astype(int), dest_port.astype(int))


split_npy_save(X_input, 20, 'X_input_split_train_n1')
np.save('./X_input_split_train_n1/Y.npy', Y)
