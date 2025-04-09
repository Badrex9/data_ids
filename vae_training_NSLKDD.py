import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Lambda, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

# ==========================
# Génération des centres de clusters
# ==========================

def random_points_on_sphere(n, d, R):
    """
    Génère n points aléatoires sur une sphère de dimension d et de rayon R.
    """
    vec = np.random.normal(size=(n, d))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    return vec * R

def update_positions(points, R, iterations=5000, learning_rate=0.01):
    """
    Optimise la répartition des points par répulsion sur une sphère de rayon R.
    Fonctionne pour toute dimension d.
    """
    for _ in range(iterations):
        dists = distance_matrix(points, points)
        np.fill_diagonal(dists, np.inf)
        
        forces = np.zeros_like(points)
        for i in range(points.shape[0]):
            diff_vectors = points[i] - points
            magnitudes = np.where(dists[i] > 0, 1 / dists[i]**2, 0)[:, np.newaxis]
            forces[i] += np.sum(diff_vectors * magnitudes, axis=0)
        
        # Normalisation et mise à jour des positions
        forces /= np.linalg.norm(forces, axis=1, keepdims=True) + 1e-6
        points += learning_rate * forces
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        points *= R
    
    return points

def find_initial_radius(n, d, alpha):
    """
    Calcule un rayon initial approximatif pour répartir les points sur une sphère de dimension d.
    """
    return alpha / (2 * np.sin(np.pi / n))

def optimize_radius(points, R, alpha, shrink_factor=0.99):
    """
    Passe d'abord par un petit rayon pour stabiliser la répartition,
    puis translate les points à R_init et optimise le rayon sans briser la contrainte alpha.
    """
    while True:
        dists = distance_matrix(points, points)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist < alpha:
            break  
        
        R *= shrink_factor
        points *= shrink_factor
    
    return R / shrink_factor, points / shrink_factor

def compute_sigma2_per_class(x, y):
    """
    Calcule sigma^2 pour chaque classe en fonction de la variance intra-classe.
    """
    unique_labels = np.unique(y)
    sigma2_per_class = {}

    x_flat = x.reshape(len(x), -1)  # Aplatir les données (si nécessaire)

    for label in unique_labels:
        class_data = x_flat[y == label]  
        class_variance = np.var(class_data, axis=0)  # Variance sur chaque dimension
        total_variance = np.sum(class_variance)  # Somme des variances sur toutes les dimensions
        sigma2_per_class[label] = total_variance  # Stockage de la variance intra-classe

    return sigma2_per_class


def compute_class_means(x, y):
    x_flat = x.reshape(len(x), -1)
    class_means = {}
    for label in np.unique(y):
        class_means[label] = np.mean(x_flat[y == label], axis=0)
    return class_means

def compute_class_distance_matrix(class_means):
    labels = sorted(class_means.keys())
    n = len(labels)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            rmse = np.sqrt(np.mean((class_means[labels[i]] - class_means[labels[j]])**2))
            dist_matrix[i, j] = rmse
            dist_matrix[j, i] = rmse
    return dist_matrix

def assign_labels_to_clusters(class_dist_matrix, cluster_centers):
    from scipy.spatial.distance import pdist, squareform
    from scipy.optimize import linear_sum_assignment

    # Distance entre centres de clusters
    cluster_dists = squareform(pdist(cluster_centers))
    
    # Inverser les distances entre classes (max - dist)
    D_max = np.max(class_dist_matrix)
    inverted_class_dists = D_max - class_dist_matrix

    # Résolution du problème d'affectation avec inversion
    cost_matrix = np.abs(cluster_dists - inverted_class_dists)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    return {i: cluster_centers[j] for i, j in zip(row_ind, col_ind)}


# ==========================
# Entraînement du VAE avec lambda
# ==========================

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.keras.layers import Input, Dense, Lambda, Reshape, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def train_vae(x, y, d, batch_size, epochs, k):

    def noiser(args):
        mean, log_var = args
        batch_size_dynamic = tf.shape(mean)[0]
        N = tf.random.normal(shape=(batch_size_dynamic, d), mean=0., stddev=1.0)
        return tf.exp(log_var / 2) * N + mean


    def kl_loss_function(label, cluster_centers, sigma2_per_class, mean, log_var):
        label = tf.cast(label, tf.int32)
        mu = mean
        sigma2_q = tf.exp(log_var)

        # On récupère le centre c_y et sigma²_p pour chaque label du batch
        c_y = tf.gather(cluster_centers, label)          # shape = (batch_size, d)
        sigma2_p = tf.gather(sigma2_per_class, label)    # shape = (batch_size, d)

        # KL divergence entre deux Gaussiennes diagonales
        kl = 0.5 * tf.reduce_sum(
            tf.math.log(sigma2_p + 1e-8) - tf.math.log(sigma2_q + 1e-8)
            + sigma2_q / (sigma2_p + 1e-8)
            + tf.square(mu - c_y) / (sigma2_p + 1e-8)
            - 1.0,
            axis=1
        )
        return kl  # shape = (batch_size,)


 

    def vae_loss(x, y, h_batch, label, cluster_centers, sigma2_per_class, mean, log_var, lambda_kl):
        x = tf.reshape(x, shape=(tf.shape(x)[0], 28 * 28))
        y = tf.reshape(y, shape=(tf.shape(y)[0], 28 * 28))

        bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction="none")
        recon_loss = tf.reduce_sum(bce_loss_fn(x, y), axis=-1)

        kl_loss = kl_loss_function(label, cluster_centers, sigma2_per_class, mean, log_var)

        total_loss = tf.cast(lambda_kl, tf.float32) * kl_loss + recon_loss
        return total_loss, recon_loss, kl_loss



    from tensorflow.keras.layers import Conv2D, Conv2DTranspose
    # --- Construction du VAE ---
    input_shape = (28, 28, 1)

    input_image = Input(shape=input_shape)
    x_enc = Conv2D(64, kernel_size=3, strides=1, padding="valid", activation="relu")(input_image)
    x_enc = Conv2D(64, kernel_size=3, strides=1, padding="valid", activation="relu")(x_enc)
    x_enc = Conv2D(64, kernel_size=3, strides=1, padding="valid", activation="relu")(x_enc)

    x_enc = Conv2D(128, kernel_size=3, padding="valid", activation="relu")(x_enc)
    x_enc = Conv2D(128, kernel_size=3, strides=1, padding="valid", activation="relu")(x_enc)
    x_enc = Conv2D(128, kernel_size=3, strides=1, padding="valid", activation="relu")(x_enc)

    x_enc = Conv2D(256, kernel_size=3, padding="valid", activation="relu")(x_enc)

    # 🔥 Passage en FCN (Fully Connected)
    x_enc = Flatten()(x_enc)

    x_enc = Dense(512, activation="relu")(x_enc)
    x_enc = Dropout(0.3)(BatchNormalization()(x_enc))

    x_enc = Dense(512, activation="relu")(x_enc)
    x_enc = Dropout(0.5)(BatchNormalization()(x_enc))

    mean = Dense(d)(x_enc)
    log_var = Dense(d)(x_enc)
    h = Lambda(noiser, name="latent_space", output_shape=(d,))([mean, log_var])

    input_decoder = Input(shape=(d,))  # d = 10
    d_dec = Dense(28 * 28 * 256, activation='relu')(input_decoder)  # Start with 256 channels
    d_dec = Reshape((28, 28, 256))(d_dec)  # Shape: (28, 28, 256)
    d_dec = Conv2D(128, kernel_size=3, padding='same', activation='relu')(d_dec)  # Reduce to 128 channels
    d_dec = Conv2D(64, kernel_size=3, padding='same', activation='relu')(d_dec)   # Reduce to 64 channels
    d_dec = Conv2D(32, kernel_size=3, padding='same', activation='relu')(d_dec)   # Reduce to 32 channels
    decoded = Conv2D(1, kernel_size=3, padding='same', activation='sigmoid', name='decoder_output')(d_dec)  # Output: (28, 28, 1)

    encoder = Model(input_image, [mean, log_var, h], name="encoder")
    decoder = Model(input_decoder, decoded, name="decoder")
    decoder.summary()
    _, _, h_encoded = encoder(input_image)
    vae_output = decoder(h_encoded)

    vae = Model(input_image, vae_output, name="vae")
    optimizer = Adam(learning_rate=5e-4)


    n_clusters = len(np.unique(y)) 
    # Calcul de sigma^2 par classe
    sigma2_per_class = compute_sigma2_per_class(x, y)
    
    sigma2 = max(sigma2_per_class.values())
    print(f"Valeur calculée de sigma2 : {sigma2:.4f}") 
    sigma = np.sqrt(sigma2)  
    alpha = k * sigma
    
    # Initialisation d'une variance vectorielle par classe (latent_dim,)
    latent_dim = d
    n_clusters = len(np.unique(y))
    # Construction d'un vecteur de variance initiale par classe
    initial_variance_vector = np.array([sigma2_per_class[c] for c in range(n_clusters)], dtype=np.float32)

    # On répète chaque sigma2 sur les latent_dim colonnes → shape (n_clusters, latent_dim)
    initial_variance_matrix = np.tile(initial_variance_vector[:, np.newaxis], (1, latent_dim))

    # Initialisation de la variable entraînable
    sigma2_per_class_var = tf.Variable(
        initial_value=initial_variance_matrix,
        trainable=True,
        dtype=tf.float32
    )

    # Calcul des distances inter-classes
    class_means = compute_class_means(x, y)
    class_dist_matrix = compute_class_distance_matrix(class_means)

    # Initialisation des centres des clusters
    R_init = find_initial_radius(n_clusters, d, alpha)
    initial_points = random_points_on_sphere(n_clusters, d, 1.0)  # Petite sphère
    optimized_points = update_positions(initial_points, 1.0)  # Optimisation sur petit rayon
    optimized_points *= R_init  # Translation vers R_init
    final_R, optimized_points = optimize_radius(optimized_points, R_init, alpha)

    
    # Associe les labels aux clusters les plus éloignés
    cluster_mapping = assign_labels_to_clusters(class_dist_matrix, optimized_points)
    ordered_centers = np.array([cluster_mapping[i] for i in sorted(cluster_mapping.keys())])
    cluster_centers = tf.constant(ordered_centers.astype(np.float32))


    # ---- Phase 1 : Estimation des statistiques X et X' (sans entraînement) ----
    X_prime_samples = []

    for j in tqdm(range(0, len(x), batch_size), desc="Calcul de X' initial", leave=False):
        batch_x = x[j:j+batch_size].astype("float32")
        _, _, h_batch = encoder(batch_x, training=False)  # Pas d'entraînement
        recon_batch = decoder(h_batch, training=False)  # Sortie initiale X'
        X_prime_samples.append(recon_batch.numpy().reshape(-1, 28 * 28))

    X_prime_samples = np.concatenate(X_prime_samples, axis=0)

    X_flat = x.reshape(-1, 28 * 28)  # Mise en forme de X

    # ---- Calcul des moyennes et variances sur X et X' ----
    E_X = np.mean(X_flat, axis=0)  # Moyenne sur chaque coordonnée j
    V_X = np.var(X_flat, axis=0)   # Variance sur chaque coordonnée j

    E_X_prime = np.mean(X_prime_samples, axis=0)  # Moyenne sur X' (chaque coordonnée j)
    V_X_prime = np.var(X_prime_samples, axis=0)   # Variance sur X' (chaque coordonnée j)

    # ---- Calcul de E[||X - X'||^2] via la formule théorique ----
    mse_theoretical = np.sum(V_X + V_X_prime + (E_X - E_X_prime) ** 2)

    print(f"MSE estimé via espérances/variances : {mse_theoretical:.4f}")
    print(f"Rmin : {final_R:.4f}")
    
    # Version avec lambda devant Dkl et pas devant R
    lambda_kl_0 = (2 * sigma2 / final_R**2) * (mse_theoretical)
    lambda_kl = lambda_kl_0

    print(f"Valeur ajustée de lambda_kl: {lambda_kl:.4f}")

    # Ajustement des centres avec lambda_kl
    # optimized_points = update_cluster_centers(optimized_points, lambda_kl, final_R)
    cluster_centers = tf.constant(optimized_points.astype(np.float32))  

    # --- Ajustement dynamique de lambda_kl ---
    def adaptive_lambda_kl(loss_history, lambda_kl_0, beta=0.01, penalty_factor=1):
        if len(loss_history) < 2:
            return lambda_kl_0  # Retourne la valeur initiale si pas assez de données
        
        # Calcul de la dérivée de la loss (différence entre deux moyennes glissantes)
        loss_diff = loss_history[-1] - loss_history[-2]
        #return lambda_kl_0 / (1 + penalty_factor * tf.exp(-beta * loss_diff))
        return 0.0001
    # --- Phase 2 : Entraînement du VAE ---
    @tf.function
    def train_step(batch_x, batch_y, lambda_kl, lambda_cycle=0.1, train_decoder_only=False):
        batch_y = tf.cast(batch_y, tf.int32)

        with tf.GradientTape(persistent=True) as tape:
            mean_val, log_var_val, h_batch = encoder(batch_x, training=True)
            recon_batch = decoder(h_batch, training=True)

            # Passage dans l'encodeur (sans backprop)
            mean_recon, _, _ = encoder(recon_batch, training=False)

            # Calcul des pertes
            total_loss, recon_loss, kl_loss = vae_loss(
                batch_x, recon_batch, h_batch, batch_y,
                cluster_centers, sigma2_per_class_var,
                mean_val, log_var_val, lambda_kl
            )

            # Ajout de la perte de cycle
            cycle_loss = tf.reduce_mean(tf.norm(mean_val - tf.stop_gradient(mean_recon), axis=1))
            total_loss += lambda_cycle * cycle_loss

        # Application des gradients
        if train_decoder_only:
            gradients = tape.gradient(total_loss, decoder.trainable_variables + [sigma2_per_class_var])
            optimizer.apply_gradients(zip(gradients, decoder.trainable_variables + [sigma2_per_class_var]))
        else:
            gradients = tape.gradient(total_loss, encoder.trainable_variables + decoder.trainable_variables + [sigma2_per_class_var])
            optimizer.apply_gradients(zip(gradients, encoder.trainable_variables + decoder.trainable_variables + [sigma2_per_class_var]))
        del tape
        return total_loss, recon_loss, kl_loss, cycle_loss

    loss_history = []
    recon_history = []
    kl_history = []
    cycle_history = []  # Ajout ici
    switch_epoch = int(epochs * 1)
    lambda_kl = 0.05
    for epoch in range(epochs):
        epoch_total_losses = []
        epoch_recon_losses = []
        epoch_kl_losses = []
        epoch_cycle_losses = []  # Ajout ici

        train_decoder_only = epoch >= switch_epoch

        for j in tqdm(range(0, len(x), batch_size), desc=f"Epoch {epoch + 1}", leave=False):
            batch_x = x[j:j+batch_size].astype("float32")
            batch_y = y[j:j+batch_size]

            total_loss, recon_loss, kl_loss, cycle_loss = train_step(
                batch_x, batch_y,
                lambda_kl=lambda_kl,
                lambda_cycle=0.01,
                train_decoder_only=train_decoder_only
            )

            epoch_total_losses.append(total_loss.numpy().mean())
            epoch_recon_losses.append(recon_loss.numpy().mean())
            epoch_kl_losses.append(kl_loss.numpy().mean())
            epoch_cycle_losses.append(cycle_loss.numpy().mean())  # Ajout ici

        loss_history.append(np.mean(epoch_total_losses))
        recon_history.append(np.mean(epoch_recon_losses))
        kl_history.append(np.mean(epoch_kl_losses))
        cycle_history.append(np.mean(epoch_cycle_losses))  # Ajout ici

        print(f"Epoch {epoch+1:2d} | Total: {loss_history[-1]:.4f} | Recon: {recon_history[-1]:.4f} | KL: {kl_history[-1]:.4f} | Cycle: {cycle_history[-1]:.4f} | Lambda_KL: {lambda_kl:.6f} | Decoder Only: {train_decoder_only}")
    return encoder, decoder, cluster_centers


from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.utils import shuffle

x_train_new = np.load(f"./mnist/mnist_x_train.npy")
y_train_new = np.load(f"./mnist/mnist_y_train.npy")

# ==========================
# Réorganiser les données pour avoir un échantillonnage équilibré
# ==========================
# Mélange stratifié des données
x_train_new_sorted, y_train_new_sorted = shuffle(x_train_new, y_train_new, random_state=42)

# Ajustement de la taille pour correspondre aux batchs
batch_size = 128
reste = x_train_new_sorted.shape[0] - (x_train_new_sorted.shape[0] % batch_size)
dataset_to_augment = x_train_new_sorted[:reste]
Y = y_train_new_sorted[:reste]

print(f"Shape of Y: {Y.shape}")
dataset_to_augment = np.reshape(dataset_to_augment, (reste, 28, 28, 1))

# ==========================
# Remap des labels avec LabelEncoder
# ==========================
encoder_label = LabelEncoder()
Y_remapped = encoder_label.fit_transform(Y)

# Affichage des labels originaux et remappés
print("Labels originaux:", Y)
print("Labels remappés:", Y_remapped)

# ==========================
# Paramètres de l'entraînement
# ==========================
d = 256  # Dimension de l'espace latent
epochs = 100
k = 40  # Facteur d'échelle entre alpha et sigma

# ==========================
# Lancement de l'entraînement
# ==========================
encoder, decoder, cluster_centers = train_vae(dataset_to_augment, Y_remapped, d, batch_size, epochs, k)

encoder.save("./models_training/encoder_training_60_zzzzz.h5")
decoder.save("./models_training/decoder_training_60_zzzzz.h5")
np.save("./models_training/cluster_training_60_zzzzz.npy", cluster_centers.numpy())

#encoder, decoder, angle_x = train_vae(dataset_to_augment, Y_remapped, d, batch_size, 100, k)

#encoder.save("./models/encoder_model_100_e.h5")
#decoder.save("./models/decoder_model_100_e.h5")
#np.save("./models/cluster_centers_100_e.npy", angle_x.numpy())

#encoder, decoder, angle_x = train_vae(dataset_to_augment, Y_remapped, d, batch_size, 200, k)

#encoder.save("./models/encoder_model_200_e.h5")
#decoder.save("./models/decoder_model_200_e.h5")
#np.save("./models/cluster_centers_200_e.npy", angle_x.numpy())