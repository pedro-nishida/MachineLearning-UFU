import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

def get_training_params():
    """Solicita parâmetros de treinamento ao usuário via terminal"""
    use_custom_epochs = input("Deseja definir um número personalizado de épocas? (s/n): ").lower() == 's'
    epochs = 100  # valor padrão
    if use_custom_epochs:
        try:
            epochs = int(input("Digite o número de épocas: "))
        except ValueError:
            print("Valor inválido. Usando o valor padrão de 2000 épocas.")
            epochs = 2000
    
    use_target_loss = input("Deseja definir um valor alvo de perda para parada antecipada? (s/n): ").lower() == 's'
    target_loss = None
    if use_target_loss:
        try:
            target_loss = float(input("Digite o valor alvo de perda: "))
        except ValueError:
            print("Valor inválido. Parada antecipada desativada.")
            target_loss = None
    
    learning_rate = 0.01  # valor padrão
    customize_lr = input("Deseja personalizar a taxa de aprendizado? (s/n): ").lower() == 's'
    if customize_lr:
        try:
            learning_rate = float(input("Digite a taxa de aprendizado: "))
        except ValueError:
            print("Valor inválido. Usando a taxa de aprendizado padrão de 0.01.")
            learning_rate = 0.01
    
    return epochs, target_loss, learning_rate

class MLP:
    def __init__(self, ninputs, nhidden, noutputs):
        # Número de camadas = número de camadas escondidas + camada de saída
        self.nlayers = len(nhidden) + 1

        # Inicialização dos pesos e vieses
        self.weights, self.biases = [], []

        # Melhorando inicialização dos pesos usando inicialização Xavier/Glorot
        scale = np.sqrt(2.0 / ninputs)
        self.weights.append(np.random.randn(nhidden[0], ninputs) * scale)
        self.biases.append(np.zeros(nhidden[0]))

        for i in range(1, len(nhidden)):
            scale = np.sqrt(2.0 / nhidden[i-1])
            self.weights.append(np.random.randn(nhidden[i], nhidden[i-1]) * scale)
            self.biases.append(np.zeros(nhidden[i]))

        scale = np.sqrt(2.0 / nhidden[-1])
        self.weights.append(np.random.randn(noutputs, nhidden[-1]) * scale)
        self.biases.append(np.zeros(noutputs))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dx_sigmoid(self, x):
        return np.exp(-x)/((1+np.exp(-x))**2)


    def forward(self, X):
        # Propagação para frente
        self.activations = [X]
        self.y = [X]

        for i in range(self.nlayers):
            activation = []
            yin = []
            w = self.weights[i]
            b = self.biases[i]

            for neuron in range(len(w)):
                soma = b[neuron]
                for j in range(len(w[neuron])):
                    soma+=w[neuron][j]*self.activations[-1][j]
                yin.append(soma)
                activation.append(self.sigmoid(soma))

            self.y.append(yin)
            self.activations.append(np.array(activation))

        resp = self.activations[-1]
        return resp


    def backward(self, X, y, learning_rate=0.01):
        dx = np.vectorize(self.dx_sigmoid)
        
        # Inicializando lista de deltas para todas as camadas
        deltas = [None] * self.nlayers
        
        # Retropropagação do erro para a camada de saída
        deltas[-1] = (self.activations[-1] - y) * dx(self.y[-1])
        
        # Propagar os erros para camadas anteriores
        for layer in range(self.nlayers - 2, -1, -1):
            delta_next = deltas[layer + 1]
            weights_next = self.weights[layer + 1]
            
            # Calcular erro para cada neurônio na camada atual
            delta = np.zeros(len(self.biases[layer]))
            
            for j in range(len(delta)):
                error_sum = 0
                for k in range(len(delta_next)):
                    error_sum += delta_next[k] * weights_next[k][j]
                delta[j] = error_sum * dx(self.y[layer + 1][j])
            
            deltas[layer] = delta
        
        # Atualizar pesos e vieses usando os deltas calculados
        for layer in range(self.nlayers):
            # Para cada neurônio na camada atual
            for j in range(len(self.weights[layer])):
                # Atualizar viés
                self.biases[layer][j] -= learning_rate * deltas[layer][j]
                
                # Para cada entrada do neurônio
                for k in range(len(self.weights[layer][j])):
                    # Atualizar peso - Note o sinal negativo aqui para descida de gradiente
                    self.weights[layer][j][k] -= learning_rate * deltas[layer][j] * self.activations[layer][k]

    def train(self, X, Y, epochs=100, learning_rate=0.01, verbose=True, report_interval=100, target_loss=None):
        losses = []
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(np.shape(X)[0]):
                x = X[i]
                y = Y[i]

                if verbose and epoch % report_interval == 0:
                    print(f"Época {epoch} ->   Forwarding", end='\r')
                output = self.forward(x)

                if verbose and epoch % report_interval == 0:
                    print(f"Época {epoch} -> Backwarding", end='\r')
                self.backward(x, y, learning_rate)

                # Calcula a perda para este exemplo
                sample_loss = -np.sum(y * np.log(output + 1e-10)) / len(y)
                epoch_loss += sample_loss

            # Média da perda para a época
            avg_loss = epoch_loss / np.shape(X)[0]
            losses.append(avg_loss)
            
            # Reduzindo a frequência dos logs para não sobrecarregar a saída
            if verbose and (epoch % report_interval == 0 or epoch == epochs - 1):
                print(f'Epoca {epoch}/{epochs}, Loss: {avg_loss:.6f}')
                
            # Salvar o melhor modelo (opcional)
            if avg_loss < best_loss:
                best_loss = avg_loss
                # Aqui poderíamos salvar os pesos se desejado
            
            # Verificar parada antecipada se target_loss foi especificado
            if target_loss is not None and avg_loss <= target_loss:
                if verbose:
                    print(f"Parada antecipada na época {epoch}. Perda alvo de {target_loss:.6f} alcançada.")
                break
        
        return losses
    
    def predict(self, X):
        """Faz previsões para um conjunto de entradas"""
        if X.ndim == 1:  # caso seja um único exemplo
            return self.forward(X)
        else:  # caso seja um batch de exemplos
            predictions = []
            for i in range(np.shape(X)[0]):
                predictions.append(self.forward(X[i]))
            return np.array(predictions)

# Exemplo de uso com os dados fornecidos
if __name__ == "__main__":
    # Dados fornecidos
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    t = np.array([-.9602, -0.5770, -0.0729, 0.3771, 0.6405, 0.6600, 0.4609, 0.1336, -0.2013, -0.4344, -0.5000])
    
    # Normalização dos dados de entrada e saída
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    
    # Normalizar também os dados de saída para melhor desempenho
    t_min = np.min(t)
    t_max = np.max(t)
    t_norm = (t - t_min) / (t_max - t_min)
    
    # Preparando os dados para o MLP
    X_train = np.array([[val] for val in x_norm])
    Y_train = np.array([[val] for val in t_norm])
    
    # Criando o MLP com configuração melhorada
    mlp = MLP(ninputs=1, nhidden=[10, 8], noutputs=1)
    
    # Obtendo parâmetros de treinamento do usuário
    print("Configuração do treinamento do MLP:")
    epochs, target_loss, learning_rate = get_training_params()
    
    # Treinando com os parâmetros definidos pelo usuário
    losses = mlp.train(X_train, Y_train, epochs=epochs, learning_rate=learning_rate, 
                      verbose=True, report_interval=500, target_loss=target_loss)
    
    # Fazendo previsões
    predictions_norm = mlp.predict(X_train)
    predictions_norm = np.array([p[0] for p in predictions_norm])
    
    # Desnormalizando as previsões para comparar com os dados originais
    predictions = predictions_norm * (t_max - t_min) + t_min
    
    # Visualizando os resultados
    plt.figure(figsize=(10, 6))
    
    # Plotando dados originais
    plt.scatter(x, t, color='blue', label='Dados Originais')
    
    # Plotando previsões
    plt.plot(x, predictions, color='red', label='Previsões do MLP')
    
    plt.title('Previsões do MLP vs Dados Originais')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    
    # Plotando a curva de perda
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.title('Curva de Perda durante o Treinamento')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.grid(True)
    
    # Verificando se os pesos estão sendo atualizados
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(mlp.weights[0].flatten(), bins=30)
    plt.title('Distribuição dos Pesos na Primeira Camada')
    
    plt.subplot(1, 2, 2)
    plt.hist(mlp.weights[-1].flatten(), bins=30)
    plt.title('Distribuição dos Pesos na Última Camada')
    
    plt.tight_layout()
    plt.show()