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

class Flux:
    #Lors de la création d'un nouveau flux, on créé un matrice vide
    def __init__(self, source_port, dest_port, packet, d_model, d_historique):
        self.sp = source_port
        self.dp = dest_port
        self.matrice = np.zeros((d_model, d_historique-1))
        for i in range(d_historique):
            self.matrice = self.matrice[:,0:d_historique-1]
            self.matrice = np.insert(self.matrice,0, packet, axis=1)

    def decalage_matriciel(self, vecteur):
        #d_historique = np.shape(self.matrice)[1]
        #On s'assure que la matrice et le vecteur sont de la bonne taille
        #assert d_model == np.shape(self.matrice)[0]
        #Décalage
        self.matrice = self.matrice[:,0:d_historique-1]
        #Ajout de la nouvelle colonne
        self.matrice = np.insert(self.matrice,0, vecteur, axis=1)
        
def flow_exists_and_append(flows, data_input, packet, source_port, dest_port):
    for flow in flows:
        sp = int(flow.sp)
        dp = int(flow.dp)
        #On teste si il existe déjà un flux connu
        if ((sp==source_port and dp==dest_port) or (sp==dest_port and dp==source_port)):
            #Si trouvé, on ajoute le nouveau vecteur en décalant la matrice
            flow.decalage_matriciel(packet)
            #On ajoute par la même occasion la matrice aux données entrante de notre modèle
            data_input.append(flow.matrice)
            return
    #Si aucun flux n'a été trouvé, on ajoute un nouveau
    flows.append(Flux(source_port, dest_port, packet, d_model, d_historique))
    #flows[len(flows)-1].decalage_matriciel(packet)
    #On ajoute la matrice correspondante à ce flux aux données entrantes
    data_input.append(flows[len(flows)-1].matrice)

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
    protocol = np.vstack(data_without_nan[1:,5])

    source_ip = np.vstack(data_without_nan[1:,1])
    dest_ip = np.vstack(data_without_nan[1:,3])
    X = data_without_nan[1:,8:]        #### On décale
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
    
    shape = np.shape(data_without_nan)
    new_colonne = np.vstack(np.zeros((shape[0],1)))
    
    X = np.concatenate((new_colonne, X), axis=1)
    X = minmax_scale(X, axis=0)
    X = np.concatenate((source_port, X), axis=1)
    X = np.concatenate((dest_port, X), axis=1)
    X = np.concatenate((ohe.values, X), axis=1)
    return X, Y, source_ip, dest_ip, protocol

def transformation_2D(X, source_ip, dest_ip):
    # Liste des flux (sauvegarde temporaire)
    flows = []
    # Données en entrée du modèle
    data_input = []
    d_model = np.shape(X)[1]
    i=0

    for raw in tqdm(X):
        #print("Raw:", raw)
        flow_exists_and_append(flows, data_input, raw, source_ip[i],dest_ip[i])
        i =i +1

    return(np.array(data_input))

def choix_donnees_entrainement_70_30(X, Y, source_ip, dest_ip):
    label_encoder = preprocessing.LabelEncoder()
    Y= label_encoder.fit_transform(Y)
    source_ip = label_encoder.fit_transform(source_ip)
    dest_ip = label_encoder.fit_transform(dest_ip)
    X_train, X_test, Y_train, Y_test, source_ip_train, source_ip_test, dest_ip_train, dest_ip_test = train_test_split(X,Y,source_ip,dest_ip,random_state=843,test_size=0.3, stratify=Y)


    return X_train, X_test, np.array(Y_train), np.array(Y_test), source_ip_train, source_ip_test, dest_ip_train, dest_ip_test

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
X_data, Y_data, source_ip_data, dest_ip_data, protocol = creation_X_Y_ip(data_frame)


#Choix des données pour l'entrainement du modèle
print("--------------------Sélection des données d'entrainement--------------------")
X, X_test, Y, Y_test, source_ip, source_ip_test, dest_ip, dest_ip_test = choix_donnees_entrainement_70_30(X_data, Y_data, source_ip_data, dest_ip_data)
print("--------------------Création des tableaux 2D pour les données entrainement--------------------")



d_model = np.shape(X)[1]
d_historique = 20
X_input_test = transformation_2D(X_test, source_ip_test, dest_ip_test)

split_npy_save(X_input_test, 10, 'X_input_split_test_zero')
