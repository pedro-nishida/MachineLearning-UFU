import numpy as np

# Carregar os dados do arquivo
data = np.loadtxt("observacoescluster.txt")

# Função para calcular a distância euclidiana entre dois pontos
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Função para inicializar os centróides aleatoriamente
def initialize_centroids(data, k):
    indices = np.random.choice(len(data), k, replace=False)
    return data[indices]

# Função para atribuir pontos aos clusters com base nos centróides
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

# Função para recalcular os centróides dos clusters
def update_centroids(clusters):
    new_centroids = [np.mean(cluster, axis=0) for cluster in clusters]
    return new_centroids

# Função para calcular o erro quadrático total (EQT)
def calculate_total_error(errors):
    return sum(errors)

# Função para verificar a convergência
def has_converged(old_centroids, new_centroids, tol=1e-4):
    return all(euclidean_distance(old, new) < tol for old, new in zip(old_centroids, new_centroids))

# Algoritmo K-means
def kmeans(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)
    eqt_values = []

    for i in range(max_iterations):
        clusters, errors = assign_to_clusters(data, centroids)
        eqt = calculate_total_error(errors)
        eqt_values.append(eqt)

        new_centroids = update_centroids(clusters)

        if has_converged(centroids, new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids, eqt_values

# Executar o K-means
k = 3  # Número de clusters desejado
clusters, centroids, eqt_values = kmeans(data, k)

# Exibir os clusters, centróides e a curva do EQT
for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1} - Centróide: {centroids[i]}")
    for point in cluster:
        print(point)
    print("\n")

# Plotar a curva do EQT
import matplotlib.pyplot as plt

plt.plot(range(len(eqt_values)), eqt_values)
plt.xlabel('Iteração')
plt.ylabel('Erro Quadrático Total (EQT)')
plt.title('Curva do EQT x Iteração')
plt.show()
