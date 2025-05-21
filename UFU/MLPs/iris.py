import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Use o backend Agg não-interativo para maior compatibilidade
import matplotlib.pyplot as plt
from classes import MLP
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Diretório para salvar as figuras
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def save_figure(fig, filename):
    """Função auxiliar para salvar figuras com tratamento de erros"""
    try:
        output_path = os.path.join(OUTPUT_DIR, filename)
        fig.savefig(output_path)
        plt.close(fig)  # Importante: fechar a figura após salvar
        print(f"Figura salva em: {output_path}")
        return True
    except Exception as e:
        print(f"ERRO ao salvar figura {filename}: {str(e)}")
        return False

def load_data_from_txt(filepath):
    """Carregar dados do arquivo de forma mais eficiente"""
    try:
        data = np.loadtxt(filepath, delimiter='\t', dtype=str)
        print(f"Dados carregados com sucesso de: {filepath}")
    except FileNotFoundError:
        # Usar caminho direto como último recurso
        direct_path = 'c:\\Users\\phfuj\\OneDrive\\Documentos\\Programacao\\AmaqRef\\ML-Reference\\MLPs\\dados.txt'
        data = np.loadtxt(direct_path, delimiter='\t', dtype=str)
        print(f"Dados carregados com sucesso de: {direct_path}")
    
    # Extrair features e labels
    features = data[:, :4].astype(float)
    labels = data[:, 4]
    
    # Informações sobre os dados
    classes, counts = np.unique(labels, return_counts=True)
    print("\nDistribuição das classes:")
    for cls, count in zip(classes, counts):
        print(f"  {cls}: {count} amostras")
        
    return features, labels

def one_hot_encode(labels):
    """Codificação one-hot mais eficiente"""
    label_dict = {
        'Iris-setosa': [1, 0, 0],
        'Iris-versicolor': [0, 1, 0],
        'Iris-virginica': [0, 0, 1]
    }
    return np.array([label_dict[label] for label in labels])

def visualize_data(X, y_raw):
    """Visualizar os dados em gráficos 2D"""
    # Criar uma nova figura explicitamente
    fig = plt.figure(figsize=(12, 5))
    
    # Definir cores para as classes
    colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'green', 'Iris-virginica': 'blue'}
    
    # Comprimento da sépala vs Largura da sépala
    plt.subplot(1, 2, 1)
    for species in np.unique(y_raw):
        indices = y_raw == species
        plt.scatter(X[indices, 0], X[indices, 1], label=species, alpha=0.7, color=colors[species])
    plt.xlabel('Comprimento da Sépala')
    plt.ylabel('Largura da Sépala')
    plt.title('Comprimento vs Largura da Sépala')
    plt.legend()
    
    # Comprimento da pétala vs Largura da pétala
    plt.subplot(1, 2, 2)
    for species in np.unique(y_raw):
        indices = y_raw == species
        plt.scatter(X[indices, 2], X[indices, 3], label=species, alpha=0.7, color=colors[species])
    plt.xlabel('Comprimento da Pétala')
    plt.ylabel('Largura da Pétala')
    plt.title('Comprimento vs Largura da Pétala')
    plt.legend()
    
    plt.tight_layout()
    
    # Usar a função auxiliar para salvar
    save_figure(fig, 'iris_visualization.png')

# Função para plotar a curva de perda
def plot_loss_curve(loss_values, filename='loss_curve.png'):
    """Plota e salva a curva de perda do treinamento"""
    fig = plt.figure(figsize=(10, 6))
    plt.plot(loss_values, color='blue')
    plt.title('Curva de Perda Durante o Treinamento')
    plt.xlabel('Época')
    plt.ylabel('Perda Média')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Destacar perda final
    plt.scatter(len(loss_values)-1, loss_values[-1], color='red', s=50, zorder=5)
    plt.annotate(f'Perda Final: {loss_values[-1]:.4f}', 
                 xy=(len(loss_values)-1, loss_values[-1]),
                 xytext=(len(loss_values)-1-len(loss_values)*0.15, loss_values[-1]*1.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                 fontsize=10)
    
    plt.tight_layout()
    save_figure(fig, filename)

def main():
    start_time = time.time()
    
    # Parâmetros configuráveis
    TRAIN_SIZE = 0.8  # 80% para treino
    BATCH_SIZE = 16   # Tamanho do batch otimizado
    LEARNING_RATE = 0.1
    EPOCHS = 5000    # Reduzir épocas para teste
    HIDDEN_LAYERS = [8, 4]
    EARLY_STOPPING_PATIENCE = 300
    
    print("=" * 50)
    print("CLASSIFICADOR MLP PARA O DATASET IRIS")
    print("=" * 50)
    
    # Carregar dados - usar diretamente o arquivo
    filepath = 'c:\\Users\\phfuj\\OneDrive\\Documentos\\Programacao\\AmaqRef\\ML-Reference\\MLPs\\dados.txt'
    print(f"Tentando carregar dados de: {filepath}")
    X, y_raw = load_data_from_txt(filepath)
    
    # Visualizar dados
    visualize_data(X, y_raw)
    
    # Normalização dos dados
    print("\nNormalizando dados...")
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    
    # Codificação one-hot
    y = one_hot_encode(y_raw)
    
    # Divisão treino/teste
    X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
        X, y, y_raw, test_size=(1-TRAIN_SIZE), random_state=42, stratify=y_raw
    )
    
    print(f"\nTotal de amostras: {len(X)}")
    print(f"Amostras de treino: {len(X_train)}")
    print(f"Amostras de teste: {len(X_test)}")
    
    # Criação e treinamento do modelo
    print("\nIniciando treinamento...")
    ninputs = X.shape[1]
    noutputs = y.shape[1]
    
    mlp = MLP(ninputs=ninputs, nhidden=HIDDEN_LAYERS, noutputs=noutputs)
    
    best_loss = float('inf')
    patience_counter = 0
    
    # Lista para armazenar valores de perda
    loss_history = []
    
    for epoch in range(EPOCHS):
        # Embaralhar dados a cada época
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        total_loss = 0
        
        # Processamento por mini-batches
        for i in range(0, len(X_train), BATCH_SIZE):
            batch_X = X_train_shuffled[i:i+BATCH_SIZE]
            batch_y = y_train_shuffled[i:i+BATCH_SIZE]
            
            # Forward pass
            outputs = mlp.forward(batch_X)
            
            # Backward pass
            mlp.backward(batch_X, batch_y, learning_rate=LEARNING_RATE)
            
            # Calcular perda média no batch
            total_loss += np.mean((outputs - batch_y) ** 2)
        
        # Calcular perda média na época
        avg_loss = total_loss * BATCH_SIZE / len(X_train)
        
        # Armazenar o valor de perda
        loss_history.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Parando treinamento antecipadamente na época {epoch} devido à falta de melhorias.")
            break
        
        if epoch % 100 == 0 or epoch == EPOCHS - 1:
            print(f"Época {epoch}, Loss: {avg_loss:.4f}")
    
    # Plotar a curva de perda após o treinamento
    plot_loss_curve(loss_history)
    
    # Avaliação do modelo
    print("\nAvaliando modelo...")
    correct = 0
    
    # Processamento por batches para avaliação mais eficiente
    predictions = []
    for i in range(0, len(X_test), BATCH_SIZE):
        batch_X = X_test[i:i+BATCH_SIZE]
        pred = mlp.forward(batch_X)
        predictions.append(pred)
    
    # Concatenar todas as previsões
    all_preds = np.vstack(predictions)
    
    # Calcular acurácia
    pred_classes = np.argmax(all_preds, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    accuracy = np.mean(pred_classes == true_classes)
    
    print(f"\nAcurácia no conjunto de teste: {accuracy:.2%}")
    
    # Matriz de confusão
    cm = confusion_matrix(true_classes, pred_classes)
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_raw), yticklabels=np.unique(y_raw))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    save_figure(fig, 'confusion_matrix.png')
    
    print(f"Tempo total de execução: {time.time() - start_time:.2f} segundos")

if __name__ == "__main__":
    main()