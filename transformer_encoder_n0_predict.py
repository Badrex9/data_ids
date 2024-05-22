import numpy as np

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

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

#print("--------------------Chargement des données train--------------------")
X_test = np.load('./X_input_split_test_n0/X_input_0.npy')
for i in range(1, 10):
    X_test = np.concatenate((X_test, np.load('./X_input_split_test_n0/X_input_'+str(i)+'.npy')))
#print("--------------------Fin du chargement des données--------------------")nction de l'entrée----------------------------------


d_output = 15 #Nombre de labels
len_x = np.shape(X_test)[0]
d_model = np.shape(X_test)[1]
d_historique = np.shape(X_test)[2] #np.shape(X_input)[1] #Longueur du vecteur d'entrée (d_model) normalement 82
num_heads = 1  #d_model % num_heads == 0, "d_model must be divisible by num_heads"
num_layers = 6 #RTIDS Nombre de répétition des encoders/decoders
d_ff = 1024 #RTIDS dimension du FFN layer
dropout = 0.5 #RTIDS
batch_size = 128 #RTIDS batch_size = 128
PATH = "./model_transformer/modele_transformer_2D.pth"
LR = 1e-5


# Defining model and training options
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

# batch_size, d_pacquet, d_model, d_historique, out_d, n_heads, n_blocks
model = MyViT(batch_size, np.shape(X_test)[1], np.shape(X_test)[1], d_historique, d_output, num_heads, num_layers, d_ff).to(device)
N_EPOCHS = 5
LR = 0.0005

model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

X_test = torch.from_numpy(X_test)

len_x = np.shape(X_test)[0]
len_without_rest = len_x - len_x%batch_size
j=0
for j in tqdm(range(0, len_without_rest, batch_size), desc=f"Predict {1}", leave=False):
    x = X_test[j:j+batch_size].transpose(1,2).to(device)
    value = model(x)
    if (j==0):
        output = value
    else:
        output = torch.cat((output, value), 0)
    print(output.size())
#On fait la vision euclidienne car le dernier batch n'est pas forcément pile de la longeur du batch voulue (plus petit)
reste = len_x%batch_size
if reste!=0:
    x = X_test[j:j+batch_size].transpose(1,2).to(device)
    value = model(x)
    if (j==0):
        output = value
    else: 
        output = torch.cat((output, value), 0)

output = output.cpu().numpy()

np.save("./Y_prediction/y_pred_transformer_2D_n0.npy", output)