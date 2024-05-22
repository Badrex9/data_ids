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
from keras.models import load_model

class Flux:
    #Lors de la création d'un nouveau flux, on créé un matrice vide
    def __init__(self, source_port, dest_port, packet, d_model, d_historique):
        self.sp = source_port
        self.dp = dest_port
        matrice_base = np.zeros((d_model, d_historique-1))
        self.matrice = np.insert(matrice_base,0, packet, axis=1)

    def decalage_matriciel(self, vecteur):
        #d_historique = np.shape(self.matrice)[1]
        #On s'assure que la matrice et le vecteur sont de la bonne taille
        #assert d_model == np.shape(self.matrice)[0]
        #Décalage
        self.matrice = self.matrice[:,0:d_historique-1]
        #Ajout de la nouvelle colonne
        self.matrice = np.insert(self.matrice,0, vecteur, axis=1)

class Flux_protocol:
    #Lors de la création d'un nouveau flux, on créé un matrice vide
    def __init__(self, protocol, packet, d_model, d_historique):
        self.protocol = protocol
        matrice_base = np.zeros((d_model, d_historique-1))
        self.matrice = np.insert(matrice_base,0, packet, axis=1)

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

def flow_exists_and_append_protocol(flows, data_input, packet, protocol):
    for flow in flows:
        pro = int(flow.protocol)
        #On teste si il existe déjà un flux connu
        if (pro==protocol):
            #Si trouvé, on ajoute le nouveau vecteur en décalant la matrice
            flow.decalage_matriciel(packet)
            #On ajoute par la même occasion la matrice aux données entrante de notre modèle
            data_input.append(flow.matrice)
            return
    #Si aucun flux n'a été trouvé, on ajoute un nouveau
    flows.append(Flux_protocol(protocol, packet, d_model, d_historique))
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
    X = minmax_scale(X, axis=0)
    X = np.concatenate((source_port, X), axis=1)
    X = np.concatenate((dest_port, X), axis=1)
    X = np.concatenate((ohe.values, X), axis=1)
    return X, Y, source_ip, dest_ip, protocol


#Choix de données de longueur nombre_données à partir d'un indice aléatoire pour un nombre_extrait
def choix_donnees_test(X, Y, source_ip, dest_ip, nombre_extrait=1000, nombre_donnees=1000):
    X_new = []
    Y_new = []
    source_ip_new = []
    dest_ip_new = []

    for i in tqdm(range(nombre_extrait)):
        indice = random.randint(0, np.shape(X)[0])
        if (indice<(np.shape(X)[0]+nombre_donnees)):
            indice = np.shape(X)[0]-nombre_donnees-2
        for j in range(nombre_donnees):
            X_new.append(X[j+indice,:])
            Y_new.append(Y[j+indice])
            source_ip_new.append(source_ip[j+indice])
            dest_ip_new.append(dest_ip[j+indice])

    label_encoder = preprocessing.LabelEncoder()
    source_ip_new = label_encoder.fit_transform(source_ip_new)
    dest_ip_new = label_encoder.fit_transform(dest_ip_new)
    Y_new= label_encoder.fit_transform(Y_new)
    return X_new, np.array(Y_new), source_ip_new, dest_ip_new

#Choix de données aléatoire dans le dataset
def choix_donnees_test3(X, Y, source_ip, dest_ip, nombre_extrait=1000, nombre_donnees=1000):

    X_new = []
    Y_new = []
    source_ip_new = []
    dest_ip_new = []

    for i in tqdm(range(nombre_extrait)):
        indice = random.randint(0, np.shape(X)[0])
        X_new.append(X[indice,:])
        Y_new.append(Y[indice])
        source_ip_new.append(source_ip[indice])
        dest_ip_new.append(dest_ip[indice])

    label_encoder = preprocessing.LabelEncoder()
    source_ip_new = label_encoder.fit_transform(source_ip_new)
    dest_ip_new = label_encoder.fit_transform(dest_ip_new)
    Y_new= label_encoder.fit_transform(Y_new)
    return X_new, np.array(Y_new), source_ip_new, dest_ip_new

#Choix de données de longueur nombre_donnees depuis l'indice nombre_extrait
def choix_donnees_test4(X, Y, source_ip, dest_ip, nombre_donnees=1000, nombre_extrait=0):

    label_encoder = preprocessing.LabelEncoder()
    Y= label_encoder.fit_transform(Y)
    X_new = []
    Y_new = []
    source_ip_new = []
    dest_ip_new = []

    for k in tqdm(range(nombre_donnees)):
      X_new.append(X[k+nombre_extrait,:])
      Y_new.append(Y[k+nombre_extrait])
      source_ip_new.append(source_ip[k+nombre_extrait])
      dest_ip_new.append(dest_ip[k+nombre_extrait])

    label_encoder = preprocessing.LabelEncoder()
    source_ip_new = label_encoder.fit_transform(source_ip_new)
    dest_ip_new = label_encoder.fit_transform(dest_ip_new)

    return X_new, np.array(Y_new), source_ip_new, dest_ip_new

#On choisit des valeurs pour chaque Label différents commencant à un indice différent (ici 20)
def choix_donnees_test_2(X, Y, source_ip, dest_ip, nombre_extrait=1000, nombre_donnes=1000):

    label_encoder = preprocessing.LabelEncoder()
    Y= label_encoder.fit_transform(Y)
    X_new = []
    Y_new = []
    source_ip_new = []
    dest_ip_new = []

    for k in tqdm(range(15)):
        i =0
        j = 0
        while (i<nombre_donnees and j<np.shape(X)[0]):
            if (Y[j]==k):
              if(20<i):
                X_new.append(X[j,:])
                Y_new.append(Y[j])
                source_ip_new.append(source_ip[j])
                dest_ip_new.append(dest_ip[j])
              i = i +1
            j = j+1


    label_encoder = preprocessing.LabelEncoder()
    source_ip_new = label_encoder.fit_transform(source_ip_new)
    dest_ip_new = label_encoder.fit_transform(dest_ip_new)
    return X_new, np.array(Y_new), source_ip_new, dest_ip_new

def choix_donnees_entrainement(X, Y, source_ip, dest_ip, nombre_donnees=1000):

    label_encoder = preprocessing.LabelEncoder()
    Y= label_encoder.fit_transform(Y)
    X_new = []
    Y_new = []
    source_ip_new = []
    dest_ip_new = []

    for k in tqdm(range(15)):
        i =0
        j = 0
        while (i<nombre_donnees and j<2824347):
            if (Y[j]==k):
                X_new.append(X[j,:])
                Y_new.append(Y[j])
                source_ip_new.append(source_ip[j])
                dest_ip_new.append(dest_ip[j])
                i = i +1
            j = j+1

    source_ip_new = label_encoder.fit_transform(source_ip_new)
    dest_ip_new = label_encoder.fit_transform(dest_ip_new)
    return X_new, np.array(Y_new), source_ip_new, dest_ip_new

def choix_donnees_entrainement_70_30(X, Y, source_ip, dest_ip):
    label_encoder = preprocessing.LabelEncoder()
    Y= label_encoder.fit_transform(Y)
    source_ip = label_encoder.fit_transform(source_ip)
    dest_ip = label_encoder.fit_transform(dest_ip)
    X_train, X_test, Y_train, Y_test, source_ip_train, source_ip_test, dest_ip_train, dest_ip_test = train_test_split(X,Y,source_ip,dest_ip,random_state=843,test_size=0.3, stratify=Y)


    return X_train, X_test, np.array(Y_train), np.array(Y_test), source_ip_train, source_ip_test, dest_ip_train, dest_ip_test

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

def transformation_2D_protocol(X,protocol):
    # Liste des flux (sauvegarde temporaire)
    flows = []
    # Données en entrée du modèle
    data_input = []
    d_model = np.shape(X)[1]
    i=0

    with tf.device('/device:GPU:0'):
        for raw in tqdm(X):
            #print("Raw:", raw)
            flow_exists_and_append_protocol(flows, data_input, raw, protocol[i])
            i =i +1

    return(np.array(data_input))

def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return np.transpose(P)

def train_model(X,Y, epochs=20):
    model = models.Sequential()
    model.add(layers.Conv2D(128, (1,2), activation='relu', input_shape=(d_model,d_historique, 1)))
    model.add(layers.AveragePooling2D((1,2)))
    model.add(layers.Conv2D(128, (1,2), activation='relu', padding='same'))
    model.add(layers.AveragePooling2D((1,2), padding='same'))
    model.add(layers.Conv2D(128, (1,2), activation='relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    #model.add(layers.Dense(128, activation='relu'))
    #model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(15))

    model.summary()
    from keras import optimizers

    model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    #print(sys.getsizeof(model))
    model.fit(X, Y, epochs=epochs, batch_size=256)#, batch_size=65536)
    model.save('./Y_prediction/model_2D_cnn_smote.h5')
    #model.save('../content/drive/MyDrive/Stage sherbrooke/Model/saved_models/model_dh_30_lr_1e-5
    return model


def print_confusion_matrix(X_input, Y, model, chemin, titre, subtitle):
    X_input = X_input.reshape(np.shape(X_input)[0],82,20,1)
    prediction = model.predict(X_input)
    y_prediction = prediction.argmax(axis=1)
    #acc_score = f1_score(Y, y_prediction)
    #print("Précision: ", acc_score*100,'%')
    Labels = ['BENIGN', 'DDoS', 'PortScan', 'Bot', 'Infiltration', 'Web Attack – Brute Force', 'Web Attack – XSS',
    'Web Attack – Sql Injection', 'FTP-Patator', 'SSH-Patator',
    'DoS slowloris', 'DoS Slowhttptest', 'DoS Hulk', 'DoS GoldenEye',
    'Heartbleed']

    #disp = ConfusionMatrixDisplay.from_predictions(Y, y_prediction, normalize='true', display_labels=Labels)
    disp = ConfusionMatrixDisplay.from_predictions(Y, y_prediction, normalize='true')
    disp.plot()
    title = titre

    plt.suptitle(title)
    #plt.title(subtitle+ ' F1 score: '+str(acc_score)+'%')
    #plt.title(subtitle+ ' F1 score: '+str(acc_score)+'%')
    plt.savefig(chemin)
    plt.show()

def split_npy_save(array, number_of_files, folder):
    file_name = 'X_input'
    i=0
    mem = 0
    len_array = len(array)
    for i in range(number_of_files):
        np.save('./'+folder+'/X_input_'+str(i)+'.npy', array[mem:int((i+1)*len_array/number_of_files)])
        mem =int((i+1)*len_array/number_of_files)
        
def main():
    print("--------------------Importation données--------------------")
    data_frame = importation_csv()
    print("--------------------Séparation des données--------------------")
    X_data, Y_data, source_ip_data, dest_ip_data = creation_X_Y_ip(data_frame)

    #Choix des données pour l'entrainement du modèle
    print("--------------------Sélection des données d'entrainement--------------------")
    X, Y, source_ip, dest_ip = choix_donnees_entrainement(X_data, Y_data, source_ip_data, dest_ip_data, nombre_donnees=2000)
    print("--------------------Création des tableaux 2D pour les données entrainement--------------------")
    d_model = np.shape(X)[1]
    d_historique = 20
    X_input = transformation_2D(X, source_ip, dest_ip)
    print("--------------------Entrainement du modèle--------------------")
    model = train_model(X_input,Y)

    #Choix des données pour la matrice de confusion
    print("--------------------Sélection des données de test--------------------")
    X_test, Y_test, source_ip_test, dest_ip_test = choix_donnees_entrainement(X_data, Y_data, source_ip_data, dest_ip_data, nombre_donnees=2000)
    print("--------------------Création des tableaux 2D pour les données test--------------------")
    X_input_test = transformation_2D(X_test, source_ip_test, dest_ip_test)
    print("--------------------Test du modèle et création de la matrice de confusion--------------------")
    print_confusion_matrix(X_input_test, Y_test, model)


d_historique = 20

print("--------------------Chargement des données train--------------------")
X_input = np.load('./X_input_split_train_no0_smote/X_input_0.npy')
d_model = np.shape(X_input)[1]
for i in range(1, 20):
    X_input = np.concatenate((X_input, np.load('./X_input_split_train_no0_smote/X_input_'+str(i)+'.npy')))
Y = np.load('./X_input_split_train/Y_no0_smote.npy')
print("--------------------Fin du chargement des données--------------------")


print("--------------------Entrainement du modèle--------------------")
model = train_model(X_input,Y, epochs=50)

