import numpy as np

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import glob
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn import preprocessing
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

def choix_donnees_entrainement_70_30(X, Y, source_ip, dest_ip):
    label_encoder = preprocessing.LabelEncoder()
    Y= label_encoder.fit_transform(Y)
    source_ip = label_encoder.fit_transform(source_ip)
    dest_ip = label_encoder.fit_transform(dest_ip)
    X_train, X_test, Y_train, Y_test, source_ip_train, source_ip_test, dest_ip_train, dest_ip_test = train_test_split(X,Y,source_ip,dest_ip,random_state=843,test_size=0.3, stratify=Y)


    return X_train, X_test, np.array(Y_train), np.array(Y_test), source_ip_train, source_ip_test, dest_ip_train, dest_ip_test


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])
    
class MyViTBlock(nn.Module):
    def __init__(self, d_model, n_heads, dff):
        super(MyViTBlock, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(d_model).double()
        self.mhsa = MyMSA(d_model, n_heads).double()
        self.norm2 = nn.LayerNorm(d_model).double()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dff).double(),
            nn.GELU(),
            nn.Linear(dff, d_model).double()
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out

class MyViT(nn.Module):
  #d_pacquet: longueur d'un pacquet en entrée
  #d_model: longueur souhaitée du modèle (inférieur ou égal à d_pacquet)
  #d_historique: nombre de séquence sauvegardé par input
  #d_out: dimension de sortie -- le nombre de labels
  def __init__(self, batch_size, d_pacquet, d_model, d_historique, out_d, n_heads, n_blocks, d_ff):
    # Super constructor
    super(MyViT, self).__init__()

    # Attributes
    self.batch_size = batch_size
    self.d_model = d_model
    self.d_historique = d_historique
    # Longueur des pacquets en entrée
    self.d_pacquet = d_pacquet
    
    # Reduction de la dimension du vecteur
    self.linear_mapper = nn.Linear(self.d_pacquet, self.d_model).double()

    # 2) Learnable classifiation token
    self.class_token = nn.Parameter(torch.rand(1, self.d_model))

    # 3) Positional embedding
    self.register_buffer('positional_embeddings', get_positional_embeddings(self.d_historique + 1, self.d_model), persistent=False)
    
    # 4) Transformer encoder blocks
    self.blocks = nn.ModuleList([MyViTBlock(self.d_model, n_heads, d_ff) for _ in range(n_blocks)])
    
    
    # 5) Classification MLPk
    self.mlp = nn.Sequential(
        nn.Linear(self.d_model, out_d).double(),
        nn.Softmax(dim=-1)
    )
    
  def forward(self, images):
    tokens = self.linear_mapper(images)

    # Adding classification token to the tokens
    tokens = torch.cat((self.class_token.expand(self.batch_size, 1, -1), tokens), dim=1)
  
    # Adding positional embedding
    out = tokens + self.positional_embeddings.repeat(self.batch_size, 1, 1)

    # Transformer Blocks
    for block in self.blocks:
        out = block(out)
        
    # Getting the classification token only
    out = out[:, 0]
    
    return self.mlp(out)


print("--------------------Importation données--------------------")
data_frame = importation_csv()
print("--------------------Séparation des données--------------------")
X_data, Y_data, source_ip_data, dest_ip_data, protocol = creation_X_Y_ip(data_frame)


#Choix des données pour l'entrainement du modèle
print("--------------------Sélection des données d'entrainement--------------------")
X_input, X_test, Y, Y_test, source_ip, source_ip_test, dest_ip, dest_ip_test = choix_donnees_entrainement_70_30(X_data, Y_data, source_ip_data, dest_ip_data)
print("--------------------Création des tableaux 2D pour les données entrainement--------------------")



d_output = 15 #Nombre de labels
len_x = np.shape(X_input)[0]
d_model = np.shape(X_input)[1]
d_historique = 1 #np.shape(X_input)[1] #Longueur du vecteur d'entrée (d_model) normalement 82
num_heads = 1  #d_model % num_heads == 0, "d_model must be divisible by num_heads"
num_layers = 6 #RTIDS Nombre de répétition des encoders/decoders
d_ff = 1024 #RTIDS dimension du FFN layer
dropout = 0.5 #RTIDS
batch_size = 1024 #RTIDS batch_size = 128
PATH = "./model_transformer/modele_transformer_1D.pth"
LR = 1e-5


# Defining model and training options
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

# batch_size, d_pacquet, d_model, d_historique, out_d, n_heads, n_blocks
model = MyViT(batch_size, np.shape(X_input)[1], np.shape(X_input)[1], d_historique, d_output, num_heads, num_layers, d_ff).to(device)
N_EPOCHS = 5
LR = 0.0005

X_input = torch.from_numpy(X_input)
Y = torch.from_numpy(Y)

# Training loop
optimizer = Adam(model.parameters(), lr=LR)
criterion = CrossEntropyLoss()
if batch_size>len_x:
    batch_size=len_x
for epoch in range(N_EPOCHS):#, desc="Training"):
    train_loss = 0.0
    len_without_rest = len_x - len_x%batch_size
    for j in tqdm(range(0, len_without_rest, batch_size), desc=f"Epoch {epoch + 1} in training", leave=False):
        x = X_input[j:j+batch_size].view(batch_size, d_model, 1).transpose(1,2).to(device)  #.view(batch_size, self.seq_len, self.d_model)
        y = Y[j:j+batch_size].to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)
        train_loss += loss.detach().cpu().item() / len_x%batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

torch.save(model.state_dict(), PATH)
