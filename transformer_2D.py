import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import math
import copy
import tqdm
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model).double()
        self.W_k = nn.Linear(d_model, d_model).double()
        self.W_v = nn.Linear(d_model, d_model).double()
        self.W_o = nn.Linear(d_model, d_model).double()
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        #if mask is not None:
            #attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)  #torch.Size([128, 1, 20, 82])
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff).double()
        self.fc2 = nn.Linear(d_ff, d_model).double()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model).double()
        self.norm2 = nn.LayerNorm(d_model).double()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model).double()
        self.norm2 = nn.LayerNorm(d_model).double()
        self.norm3 = nn.LayerNorm(d_model).double()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, enc_output, src_mask):
        attn_output = self.self_attn(enc_output, enc_output, enc_output, src_mask)
        x = enc_output
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout, d_output, seq_len):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(100, d_model)
        #Création des encoders et decoder au nombre de num_layers chacun
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_heads = num_heads
        #Creation de la couche linéaire finale
        self.fc = nn.Linear(d_model, d_output).double()
        #Creation du dropout
        #self.dropout = nn.Dropout(dropout) -------------A voir-------------

    def generate_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(3)
        return src_mask

    def forward(self, src):
        src_mask = self.generate_mask(src)
        batch_size, d_model, _, seq_len, _ = src_mask.size()
        #src_mask = src_mask.reshape(batch_size, self.num_heads, self.num_heads, self.seq_len, self.d_model)
        #src_mask = src_mask.view(batch_size, self.num_heads, self.d_model, self.seq_len)
        #src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src))) pas besoin deja embedded; on change un peu la fonction

        #-----------------------------Utilisation de dropout-----------------------------
        #src_embedded = self.dropout(src)
        #-----------------------------Utilisation de dropout-----------------------------
        
        #On créé des valeur inutile pour garder la structure de base si on besoin d'une adaptation avec dropout
        src_embedded = src

        #Couche des N encodeurs
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        
        #Couche des N decodeurs
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(enc_output, src_mask)
        
        #Sortie linéaire
        output = self.fc(dec_output)
        return output
    
    def one_epoch(self, j, X_input, Y, batch_size, optimizer, criterion, device):
        input = X_input[j:j+batch_size].transpose(1,2).to(device)  #.view(batch_size, self.seq_len, self.d_model)
        labels = Y[j:j+batch_size].view(batch_size)
        optimizer.zero_grad()
        
        output = self(input)
        loss = criterion(F.softmax(output[:,0,:], dim=1), labels)
        loss.backward()

        optimizer.step()
        loss_value = loss.item()
        print(loss_value)
        return loss_value
    
    def train_model(self, X_input, Y, batch_size, num_epochs, device):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=1e-5)  #RTIDS
        len_x = np.shape(X_input)[0] 
        if batch_size>len_x:
            batch_size=len_x
        for epoch in range(num_epochs):
            self.train(True)
            running_loss = 0.
            len_without_rest = len_x - len_x%batch_size
            for j in range(0, len_without_rest, batch_size):
                running_loss += self.one_epoch(j, X_input, Y, batch_size, optimizer, criterion, device)
            #On fait la vision euclidienne car le dernier batch n'est pas forcément pile de la longeur du batch voulue (plus petit)
            if len_x%batch_size!=0:
                running_loss += self.one_epoch(j, X_input, Y, len_x%batch_size, optimizer, criterion, device)
            print(f"Epoch: {epoch+1}, Loss: {running_loss}")
    
    def predict(self, X_test, batch_size):
        len_x = np.shape(X_test)[0]
        len_without_rest = len_x - len_x%batch_size

        j=0
        for j in range(0, len_without_rest, batch_size):
            value = self(X_test[j:j+batch_size].view(batch_size, self.seq_len, self.d_model))
            if (j==0):
                output = torch.argmax(F.softmax(value)[:,0,:], dim = 1)
            else:
                output = torch.cat((output, torch.argmax(F.softmax(value)[:,0,:], dim = 1)), 0)
        #On fait la vision euclidienne car le dernier batch n'est pas forcément pile de la longeur du batch voulue (plus petit)
        reste = len_x%batch_size
        if reste!=0:
            value = self(X_test[j:j+reste].view(reste, self.seq_len, self.d_model))
            if (j==0):
                output = torch.argmax(F.softmax(value)[:,0,:], dim = 1)
            else: 
                output = torch.cat((output, torch.argmax(F.softmax(value)[:,0,:], dim = 1)), 0)
        return output


print("--------------------Chargement des données train--------------------")
X_input = np.load('./X_input_split_train_n0/X_input_0.npy')
d_model = np.shape(X_input)[1]
for i in range(1, 20):
    X_input = np.concatenate((X_input, np.load('./X_input_split_train_n0/X_input_'+str(i)+'.npy')))
Y = np.load('./X_input_split_train/Y.npy')
print("--------------------Fin du chargement des données--------------------")


d_output = 15 #Nombre de labels
d_model = np.shape(X_input)[1]
seq_len =  np.shape(X_input)[2]#np.shape(X_input)[1] #Longueur du vecteur d'entrée (d_model) normalement 82
num_heads = 2  #d_model % num_heads == 0, "d_model must be divisible by num_heads"
num_layers = 6 #RTIDS Nombre de répétition des encoders/decoders
d_ff = 1024 #RTIDS dimension du FFN layer
dropout = 0.5 #RTIDS
batch_size = 128 #RTIDS batch_size = 128
epochs = 50
PATH = "./model_transformer/modele_transformer_2D.pth"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


X_input = torch.from_numpy(X_input)
Y = torch.from_numpy(Y)


transformer = Transformer(d_model, num_heads, num_layers, d_ff, dropout, d_output, seq_len)
transformer.to(device)
transformer.train_model(X_input, Y, batch_size, epochs, device)


torch.save(transformer.state_dict(), PATH)