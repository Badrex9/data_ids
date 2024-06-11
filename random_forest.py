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
import joblib

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
    X = data_without_nan[1:,7:]        #### On décale
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
    
    X = minmax_scale(X, axis=0)
    X = np.concatenate((source_port, X), axis=1)
    X = np.concatenate((dest_port, X), axis=1)
    X = np.concatenate((ohe.values, X), axis=1)
    return X, Y, source_ip, dest_ip, protocol


def choix_donnees_entrainement_70_30(X, Y, source_ip, dest_ip):
    label_encoder = preprocessing.LabelEncoder()
    Y= label_encoder.fit_transform(Y)
    source_ip = label_encoder.fit_transform(source_ip)
    dest_ip = label_encoder.fit_transform(dest_ip)
    X_train, X_test, Y_train, Y_test, source_ip_train, source_ip_test, dest_ip_train, dest_ip_test = train_test_split(X,Y,source_ip,dest_ip,random_state=843,test_size=0.3, stratify=Y)


    return X_train, X_test, np.array(Y_train), np.array(Y_test), source_ip_train, source_ip_test, dest_ip_train, dest_ip_test

print("--------------------Importation données--------------------")
data_frame = importation_csv()
print("--------------------Séparation des données--------------------")
X_data, Y_data, source_ip_data, dest_ip_data, protocol = creation_X_Y_ip(data_frame)


#Choix des données pour l'entrainement du modèle
print("--------------------Sélection des données d'entrainement--------------------")
X, X_test, Y, Y_test, source_ip, source_ip_test, dest_ip, dest_ip_test = choix_donnees_entrainement_70_30(X_data, Y_data, source_ip_data, dest_ip_data)
print("--------------------Création des tableaux 2D pour les données entrainement--------------------")

# import Random Forest classifier
from sklearn.ensemble import RandomForestClassifier

# instantiate the classifier 
rfc = RandomForestClassifier(random_state=0)

Y_new = []

for label in Y:
    if label==12 or label==13 or label==14:
        Y_new.append(12)
    else:
        Y_new.append(label)

Y = Y_new

# fit the model
rfc.fit(X, Y)

Y_new = []

for label in Y_test:
    if label==12 or label==13 or label==14:
        Y_new.append(12)
    else:
        Y_new.append(label)

Y_test = Y_new

joblib.dump(rfc, "./Y_prediction/random_forest_WEB_ATTACK.joblib")

# Predict the Test set results
y_pred = rfc.predict(X_test)

np.save('./Y_prediction/Y_random_forest_WEB_ATTACK.npy', y_pred)