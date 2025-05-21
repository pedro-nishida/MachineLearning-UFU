import numpy as np
import matplotlib.pyplot as plt
import os

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Função para inicializar os centróides aleatoriamente
def initialize_centroids(data, k):
    indices = np.random.choice(len(data), k, replace=False)
    return data[indices]

def initialize_centroids_pp(data, k):
    # Primeiro centróide aleatório
    centroids = [data[np.random.randint(len(data))]]
    centroids_array = np.array(centroids)
    
    for _ in range(1, k):
        # Calcular matriz de distâncias entre todos os pontos e centroides atuais
        # shape: [n_points, n_centroids]
        dist_matrix = np.array([[euclidean_distance(point, centroid) 
                                for centroid in centroids_array] 
                                for point in data])
        
        # Pegar a menor distância para cada ponto
        min_distances = np.min(dist_matrix, axis=1)
        
        # Calcular probabilidades
        probabilities = min_distances / np.sum(min_distances)
        
        # Escolher próximo centróide
        next_centroid_idx = np.random.choice(len(data), p=probabilities)
        centroids.append(data[next_centroid_idx])
        centroids_array = np.array(centroids)
        
    return centroids_array

def assign_to_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    errors = []

    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append(point)
        error = distances[cluster_index] ** 2
        errors.append(error)

    return clusters, errors

def assign_to_clusters_vectorized(data, centroids):
    # Calcula todas as distâncias de uma vez (shape: [n_points, n_centroids])
    distances = np.array([[euclidean_distance(point, centroid) for centroid in centroids] for point in data])
    
    # Encontra o índice do centróide mais próximo para cada ponto
    cluster_indices = np.argmin(distances, axis=1)
    
    # Inicializa clusters
    clusters = [[] for _ in range(len(centroids))]
    errors = []
    
    # Atribui pontos aos clusters
    for i, point in enumerate(data):
        cluster_idx = cluster_indices[i]
        clusters[cluster_idx].append(point)
        errors.append(distances[i, cluster_idx] ** 2)
    
    return clusters, errors

def update_centroids(clusters, data, previous_centroids):
    new_centroids = []
    
    for i, cluster in enumerate(clusters):
        if cluster:  # Se o cluster não estiver vazio
            new_centroids.append(np.mean(cluster, axis=0))
        else:  # Se o cluster estiver vazio
            # Estratégia 1: Manter o centróide anterior
            new_centroids.append(previous_centroids[i])
            # Alternativa: Escolher um ponto aleatório
            # new_centroids.append(data[np.random.randint(len(data))])
    
    return np.array(new_centroids)

def calculate_total_error(errors):
    return sum(errors)

def has_converged(old_centroids, new_centroids, tol=1e-4):
    # Calcular distâncias entre centróides antigos e novos de uma vez
    distances = np.array([euclidean_distance(old, new) 
                         for old, new in zip(old_centroids, new_centroids)])
    return np.all(distances < tol)

# Algoritmo K-means
def kmeans(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)
    eqt_values = []

    for i in range(max_iterations):
        clusters, errors = assign_to_clusters(data, centroids)
        eqt = calculate_total_error(errors)
        eqt_values.append(eqt)

        new_centroids = update_centroids(clusters, data, centroids)

        if has_converged(centroids, new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids, eqt_values

def kmeanspp(data, k, max_iterations=100):
    centroids = initialize_centroids_pp(data, k)
    eqt_values = []

    for i in range(max_iterations):
        clusters, errors = assign_to_clusters(data, centroids)
        eqt = calculate_total_error(errors)
        eqt_values.append(eqt)

        new_centroids = update_centroids(clusters, data, centroids)

        if has_converged(centroids, new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids, eqt_values

def plot_kmeans_results(clusters, centroids, eqt_values, ax_clusters, ax_eqt, title, colors):
    # Plotar clusters e centroides
    for i, cluster in enumerate(clusters):
        if not cluster:  # Pular clusters vazios
            continue
        cluster_array = np.array(cluster)
        # Plotar pontos do cluster
        ax_clusters.scatter(cluster_array[:, 0], cluster_array[:, 1], 
                            c=colors[i % len(colors)], 
                            label=f'Cluster {i + 1}', alpha=0.5)
        # Marcar centróide
        ax_clusters.scatter(centroids[i][0], centroids[i][1], 
                            marker='o', s=200, c=colors[i % len(colors)], 
                            edgecolors='black')
    
    ax_clusters.set_xlabel('Feature 1')
    ax_clusters.set_ylabel('Feature 2')
    ax_clusters.set_title(f'Clusters e Centróides: {title}')
    ax_clusters.legend()
    
    # Plotar curva EQT
    ax_eqt.plot(range(len(eqt_values)), eqt_values)
    ax_eqt.set_xlabel('Iteração')
    ax_eqt.set_ylabel(f'Erro Quadrático Total (EQT) {title}')
    ax_eqt.set_title('Curva do EQT x Iteração')

# Número de Clusters
k = 4

# Plotar os dados de cada cluster com cores diferentes e marcando as centroides
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
fig, ((axK1, axK2), (axKPP1, axKPP2)) = plt.subplots(2, 2, constrained_layout=True)

# Carregar os dados do arquivo (especifique o caminho correto)
# Substitua pelo caminho onde o arquivo realmente está
file_path = os.path.join(os.path.dirname(__file__), "observacoescluster.txt")
# Ou use o caminho completo: file_path = "C:\\caminho\\completo\\observacoescluster.txt"

try:
    data = np.loadtxt(file_path)
except FileNotFoundError:
    print(f"Arquivo não encontrado em: {file_path}")
    print("Gerando dados sintéticos...")
    # Código de geração de dados aqui...

# Executar o K-means com inicialização k-means
clusters, centroids, eqt_values = kmeans(data, k)

# Exibir os clusters, centróides e a curva do EQT
for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1} - Centróide: {centroids[i]}")
    for point in cluster:
        print(point)
    print("\n")

# Plotar os resultados do K-means
plot_kmeans_results(clusters, centroids, eqt_values, axK1, axK2, "K-Means", colors)

# Executar o K-means com inicialização k-means++
clusters, centroids, eqt_values = kmeanspp(data, k)

# Exibir os clusters, centróides e a curva do EQT
for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1} - Centróide: {centroids[i]}")
    for point in cluster:
        print(point)
    print("\n")

# Plotar os resultados do K-means++
plot_kmeans_results(clusters, centroids, eqt_values, axKPP1, axKPP2, "K-Means++", colors)

plt.show()
fig.savefig("Graphs.png")