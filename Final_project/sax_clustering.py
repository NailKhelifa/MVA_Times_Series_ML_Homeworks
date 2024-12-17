import numpy as np
import pandas as pd 
from distances import MINDIST, TRENDIST


def k_means_sax(sax, max_iter, num_cluster):
    """
    Implémente l'algorithme K-Means adapté pour des observations sous forme de chaînes.

    Parameters:
    sax (SYMBOLS): comme rendu par SYMBOLS()
    max_iter (int): Nombre maximum d'itérations.
    num_cluster (int): Nombre de clusters à former.

    Returns:
    tuple: Un tuple contenant :
        - Les indices des centres finaux des clusters (list).
        - Les labels des clusters pour chaque observation (np.ndarray de taille (n,)).
        - La distance intra-cluster moyenne (float).
        - La distance inter-cluster moyenne (float).
    """
    data = sax.symbolized_x_train.iloc[1,:]

    # Choix de la mesure de distance
    if sax.method == "TSAX":
        dist = TRENDIST(sax.alphabet_size, sax.train_ts_length, sax.angle_breakpoint_alphabet_size)
    else:
        dist = MINDIST(sax.alphabet_size, sax.train_ts_length)

    # Étape 1 : Initialisation
    num_samples = data.shape[0]  # nombre d'observations
    np.random.seed(42)  # Pour la reproductibilité
    initial_indices = np.random.choice(num_samples, num_cluster, replace=False)
    centroids = initial_indices  # Les centroïdes sont initialisés par des indices

    # Initialiser les labels des clusters (0, 1, ..., num_cluster-1)
    labels = np.zeros(num_samples, dtype=int)

    # Boucle principale de l'algorithme K-Means
    for iteration in range(max_iter):
        # Étape 2 : Assignation des observations aux clusters les plus proches
        for i in range(num_samples):
            distances = []
            for centroid_idx in centroids:
                if sax.method == "TSAX":
                    distances.append(dist.tsax_mindist(data.iloc[i, :][0], data.iloc[centroid_idx, :][0]))
                else:
                    distances.append(dist.mindist(data.iloc[i, :][0], data.iloc[centroid_idx, :][0]))
            # Trouver le centroïde le plus proche et assigner l'observation à ce cluster
            labels[i] = np.argmin(distances)

        # Étape 3 : Mise à jour des centroïdes
        new_centroids = []
        for k in range(num_cluster):
            # Extraire les indices des observations appartenant au cluster k
            cluster_indices = np.where(labels == k)[0]
            if len(cluster_indices) > 0:
                # Trouver le point le plus central dans le cluster
                min_distance_sum = float("inf")
                central_index = cluster_indices[0]
                for idx in cluster_indices:
                    # Calculer la somme des distances de ce point à tous les autres du cluster
                    distance_sum = 0
                    for other_idx in cluster_indices:
                        if sax.method == "TSAX":
                            distance_sum += dist.tsax_mindist(data.iloc[idx], data.iloc[other_idx])
                        else:
                            distance_sum += dist.mindist(data.iloc[idx], data.iloc[other_idx])
                    # Mettre à jour le point central si une plus petite somme est trouvée
                    if distance_sum < min_distance_sum:
                        min_distance_sum = distance_sum
                        central_index = idx
                new_centroids.append(central_index)
            else:
                # Si un cluster est vide, réinitialiser son centroïde de manière aléatoire
                new_centroids.append(np.random.choice(num_samples))

        # Vérifier la convergence
        if np.array_equal(centroids, new_centroids):
            print(f"Convergence atteinte après {iteration + 1} itérations.")
            break
        
        # Mettre à jour les centroïdes pour la prochaine itération
        centroids = new_centroids

    # Calcul de la distance intra-cluster moyenne
    intra_cluster_distances = []
    for k in range(num_cluster):
        cluster_indices = np.where(labels == k)[0]
        for idx in cluster_indices:
            if sax.method == "TSAX":
                intra_cluster_distances.append(dist.tsax_mindist(data.iloc[idx], data.iloc[centroids[k]]))
            else:
                intra_cluster_distances.append(dist.mindist(data.iloc[idx], data.iloc[centroids[k]]))
    intra_cluster_mean_distance = np.mean(intra_cluster_distances) if intra_cluster_distances else 0.0

    # Calcul de la distance inter-cluster moyenne
    inter_cluster_distances = []
    for i in range(num_cluster):
        for j in range(i + 1, num_cluster):
            if sax.method == "TSAX":
                inter_cluster_distances.append(
                    dist.tsax_mindist(data.iloc[centroids[i]], data.iloc[centroids[j]])
                )
            else:
                inter_cluster_distances.append(
                    dist.mindist(data.iloc[centroids[i]], data.iloc[centroids[j]])
                )
    inter_cluster_mean_distance = np.mean(inter_cluster_distances) if inter_cluster_distances else 0.0

    return centroids, labels, intra_cluster_mean_distance, inter_cluster_mean_distance



def calculate_distance_matrix(sax, clusters):
    """
    Calcule la matrice des distances entre les clusters actuels.
    """
    if sax.method == "TSAX":
        dist = TRENDIST(sax.alphabet_size, sax.train_ts_length, sax.angle_breakpoint_alphabet_size)
    else: 
        dist = MINDIST(sax.alphabet_size, sax.train_ts_length)

    num_clusters = len(clusters)
    distance_matrix = np.full((num_clusters, num_clusters), np.inf)

    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            # Calculer la distance entre les centroïdes des clusters i et j
            if sax.method == "TSAX":
                d = dist.tsax_mindist(clusters[i]['representative'], clusters[j]['representative'])
            else: 
                d = dist.mindist(clusters[i]['representative'], clusters[j]['representative'])

            distance_matrix[i, j] = d
            distance_matrix[j, i] = d

    return distance_matrix

# Fonction principale pour le clustering hiérarchique
def hierarchical_clustering(sax, num_cluster):
    """
    Implémente le clustering hiérarchique agglomératif from scratch.
    
    Paramètres :
    - data : pd.DataFrame (n, p) : les données d'entrée.
    - max_iter : int : le nombre maximum d'itérations avant d'arrêter.
    - num_cluster : int : le nombre de clusters final souhaité.

    Retourne :
    - clusters : Liste contenant les clusters finaux.
    - linkage_matrix : np.ndarray : matrice des liens pour suivre les fusions de clusters.
    """

    data = sax.symbolized_x_train
    num_samples = data.shape[0]

    # on aura beosin de la métrique pour recalculr les représentants de chaque cluster fusioné 
    if sax.method == "TSAX":
        D = TRENDIST(sax.alphabet_size, sax.train_ts_length, sax.angle_breakpoint_alphabet_size)
    else: 
        D = MINDIST(sax.alphabet_size, sax.train_ts_length)
    
    # Initialisation : chaque point est un cluster
    clusters = []
    linkage_matrix = []  # Liste pour stocker la matrice des liens
    current_cluster_id = num_samples  # On commence à partir de num_samples pour les nouveaux clusters

    for i in range(num_samples):
        clusters.append({
            'id': i,  # Identifiant unique du cluster
            'points': [data.iloc[i, :][0]],  # Liste des points dans le cluster
            'representative': data.iloc[i, :][0]  # Représentant initial = le point lui-même
        })

    iteration = 0
    
    while len(clusters) > num_cluster:
        # Étape 1 : Calculer la matrice des distances
        distance_matrix = calculate_distance_matrix(sax, clusters)

        # Étape 2 : Trouver les clusters les plus proches à fusionner
        i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)

        # Récupérer les distances minimales et les IDs des clusters fusionnés
        dist = distance_matrix[i, j]
        id_i, id_j = clusters[i]['id'], clusters[j]['id']

        # Ajouter à la linkage_matrix avec une vérification
        if len(clusters[i]['points']) > 0 and len(clusters[j]['points']) > 0:
            linkage_matrix.append([id_i, id_j, dist, len(clusters[i]['points']) + len(clusters[j]['points'])])
        else:
            raise ValueError(f"Fusion invalide, l'un des clusters {id_i} ou {id_j} est vide ou non formé correctement.")

        # Étape 3 : Fusionner les deux clusters trouvés
        merged_points = clusters[i]['points'] + clusters[j]['points']

        if sax.method == "TSAX":
            rep = [sum(D.tsax_mindist(p, q) for q in merged_points) for p in merged_points]
        else: 
            rep = [sum(D.mindist(p, q) for q in merged_points) for p in merged_points]

        merged_cluster = {
            'id': current_cluster_id,  # Nouveau cluster avec un identifiant unique
            'points': merged_points,  # Fusion des points
            'representative': merged_points[np.argmin(rep)]  # Point médian
        }

        # Supprimer les clusters i et j et ajouter le nouveau cluster
        clusters.pop(max(i, j))  # Supprimer le plus grand index d'abord pour éviter les décalages
        clusters.pop(min(i, j))
        clusters.append(merged_cluster)

        # Incrémenter l'ID pour le prochain cluster fusionné
        current_cluster_id += 1
        
        # Incrémenter le compteur d'itérations
        iteration += 1

    # Convertir linkage_matrix en tableau numpy
    linkage_matrix = np.array(linkage_matrix)

    return clusters, linkage_matrix