import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from evolution_graph import EvolutionGraph
import time

try:
    import tensorflow as tf
    TF_AVAILABLE = True
    print("TensorFlow disponível, versão:", tf.__version__)
    print("GPUs Disponíveis:", tf.config.list_physical_devices('GPU'))
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow não encontrado. Usando implementação NumPy.")

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

    def train(self, X, Y, epochs=100, learning_rate=0.01, verbose=True, report_interval=100, target_loss=None, show_graph=False):
        losses = []
        best_loss = float('inf')
        
        # Criar e iniciar o gráfico de evolução, se solicitado
        graph = None
        if show_graph:
            graph = EvolutionGraph(max_score=1.0)  # Valor máximo definido para 1.0 (ajuste conforme necessário)
            graph.run_in_thread()
        
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
            
            # Atualizar o gráfico de evolução, se estiver sendo usado
            if show_graph and graph:
                graph.add_score(avg_loss)
            
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
        
        # Manter o gráfico aberto por um momento para visualização
        if show_graph and graph:
            # Aguardar para não fechar o gráfico imediatamente
            print("Gráfico de treinamento aberto. Fechará automaticamente em alguns segundos...")
            time.sleep(5)  # Espera 5 segundos
        
        return losses, graph

    def predict(self, X):
        """Faz previsões para um conjunto de entradas"""
        if X.ndim == 1:  # caso seja um único exemplo
            return self.forward(X)
        else:  # caso seja um batch de exemplos
            predictions = []
            for i in range(np.shape(X)[0]):
                predictions.append(self.forward(X[i]))
            return np.array(predictions)


class MLPVisualizer:
    def __init__(self, mlp):
        self.mlp = mlp
        self.G = nx.DiGraph()
        self.pos = {}
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.create_graph()
    
    def create_graph(self):
        """Cria o grafo da rede neural"""
        # Determina o tamanho de cada camada corretamente
        layer_sizes = []
        
        # Camada de entrada - mesmo tamanho que a primeira dimensão dos pesos
        input_size = len(self.mlp.weights[0][0])
        layer_sizes.append(input_size)
        
        # Camadas ocultas e de saída
        for i in range(len(self.mlp.weights)):
            layer_sizes.append(len(self.mlp.weights[i]))
        
        print(f"Estrutura da rede: {layer_sizes} neurônios por camada")
        
        # Adiciona nós ao grafo
        node_id = 0
        x_spacing = 4  # Aumentado espaçamento horizontal para melhor visualização
        y_spacing = 2  # Aumentado espaçamento vertical
        
        # Dicionário para mapear (camada, índice_neurônio) para node_id
        self.layer_node_map = {}
        
        for layer_idx, num_neurons in enumerate(layer_sizes):
            layer_y_offset = -(num_neurons - 1) * y_spacing / 2  # Centralizar verticalmente
            
            for neuron_idx in range(num_neurons):
                # Adiciona o nó com atributos de camada e índice
                self.G.add_node(node_id, 
                               layer=layer_idx, 
                               neuron_idx=neuron_idx,
                               label=f"L{layer_idx}-N{neuron_idx}")
                
                # Posiciona o nó no espaço
                y_pos = layer_y_offset + neuron_idx * y_spacing
                self.pos[node_id] = (layer_idx * x_spacing, y_pos)
                
                # Mapeia (camada, índice_neurônio) para node_id
                self.layer_node_map[(layer_idx, neuron_idx)] = node_id
                
                node_id += 1
        
        # Adiciona arestas com pesos iniciais (completamente conectado entre camadas)
        for layer in range(len(self.mlp.weights)):
            for j in range(len(self.mlp.weights[layer])):  # Para cada neurônio na camada atual
                for i in range(len(self.mlp.weights[layer][j])):  # Para cada conexão com a camada anterior
                    start_node = self.layer_node_map.get((layer, i))
                    end_node = self.layer_node_map.get((layer+1, j))
                    
                    if start_node is not None and end_node is not None:
                        weight = self.mlp.weights[layer][j][i]
                        self.G.add_edge(start_node, end_node, weight=weight)
    
    def update_graph(self, frame):
        self.ax.clear()
        # Obtém as ativações para o frame atual ou usa o último frame se estiver fora do alcance
        activation_step = self.mlp.activations[min(frame, len(self.mlp.activations)-1)]
        
        # Mapeia as ativações para cada nó usando o dicionário de mapeamento
        node_values = {}
        
        # Para cada camada no modelo
        for layer_idx, layer_act in enumerate(activation_step):
            # Garante que estamos trabalhando com array numpy
            if not isinstance(layer_act, np.ndarray):
                if isinstance(layer_act, list):
                    layer_act = np.array(layer_act)
                else:
                    layer_act = np.array([layer_act])
            
            # Mapeia ativações para nós
            for i, val in enumerate(layer_act):
                node_id = self.layer_node_map.get((layer_idx, i))
                if node_id is not None:
                    node_values[node_id] = float(val)
        
        # Gera cores e tamanhos para todos os nós
        colors = []
        sizes = []
        labels = {}
        
        for node in self.G.nodes():
            val = node_values.get(node, 0.0)
            colors.append(plt.cm.viridis(val))
            sizes.append(300 + 700 * abs(val))  # Tamanho aumentado para melhor visualização
            
            # Adiciona rótulos com valores de ativação
            layer = self.G.nodes[node]['layer']
            neuron_idx = self.G.nodes[node]['neuron_idx']
            labels[node] = f"{val:.2f}"
        
        # Desenha o grafo com as cores e tamanhos mapeados
        nx.draw(self.G, self.pos, ax=self.ax, 
                node_color=colors, 
                node_size=sizes, 
                with_labels=True,
                labels=labels,
                font_size=9,
                font_color='black',
                font_weight='bold',
                edge_color='gray',
                width=1.0,
                alpha=0.8)
        
        # Adicionar título e informações
        self.ax.set_title(f'Propagação na Rede Neural - Frame {frame}')
        
        # Corrigir a criação da informação de estrutura para lidar com escalares
        layer_sizes = []
        for act in activation_step:
            if isinstance(act, (list, np.ndarray)):
                layer_sizes.append(str(len(act)))
            else:
                layer_sizes.append("1")  # Para valores escalares
                
        layer_info = " → ".join(layer_sizes)
        self.ax.text(0.5, -0.05, f"Estrutura da rede: {layer_info}", 
                  transform=self.ax.transAxes, 
                  horizontalalignment='center',
                  fontsize=12)
   
    def animate(self):
        ani = animation.FuncAnimation(self.fig, self.update_graph, frames=len(self.mlp.activations), interval=500)
        plt.show()

class MLPTF:
    """Implementação de MLP usando TensorFlow para aproveitar aceleração de GPU"""
    
    def __init__(self, ninputs, nhidden, noutputs):
        """
        Inicializa uma rede MLP usando TensorFlow
        
        Args:
            ninputs (int): Número de neurônios na camada de entrada
            nhidden (list): Lista com o número de neurônios em cada camada escondida
            noutputs (int): Número de neurônios na camada de saída
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow não está instalado. Use 'pip install tensorflow'.")
        
        # Salvando a estrutura para referência
        self.ninputs = ninputs
        self.nhidden = nhidden
        self.noutputs = noutputs
        self.nlayers = len(nhidden) + 1
        
        # Criando um modelo sequencial, seguindo a recomendação de usar Input como primeira camada
        self.model = tf.keras.Sequential()
        
        # Definindo a entrada explicitamente
        self.model.add(tf.keras.layers.Input(shape=(ninputs,)))
        
        # Adicionando a primeira camada escondida
        self.model.add(tf.keras.layers.Dense(
            nhidden[0], 
            activation='sigmoid',
            kernel_initializer=tf.keras.initializers.GlorotNormal()
        ))
        
        # Adicionando camadas escondidas adicionais
        for i in range(1, len(nhidden)):
            self.model.add(tf.keras.layers.Dense(
                nhidden[i], 
                activation='sigmoid',
                kernel_initializer=tf.keras.initializers.GlorotNormal()
            ))
        
        # Adicionando camada de saída
        self.model.add(tf.keras.layers.Dense(
            noutputs, 
            activation='sigmoid',
            kernel_initializer=tf.keras.initializers.GlorotNormal()
        ))
        
        # Compilando o modelo
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Construir o modelo com um exemplo de entrada
        self.model.build((None, ninputs))
        
        # Lista para armazenar ativações
        self.activations = []
        
        # Os modelos auxiliares serão criados no primeiro forward pass
        self.layer_outputs = None
        
    def forward(self, X):
        """
        Realiza a propagação para frente (forward pass)
        
        Args:
            X: Dados de entrada (deve ser um array numpy)
            
        Returns:
            np.array: Saída da rede neural
        """
        # Converter entrada para tensor se necessário
        if isinstance(X, np.ndarray) and X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Criar modelos auxiliares para extração de ativações, se ainda não existirem
        if self.layer_outputs is None:
            self.layer_outputs = []
            # A primeira "camada" será apenas a entrada
            
            # Para cada camada no modelo, criar um modelo que retorna sua saída
            for i in range(len(self.model.layers)):
                layer_model = tf.keras.Model(
                    inputs=self.model.input,
                    outputs=self.model.layers[i].output
                )
                self.layer_outputs.append(layer_model)
        
        # Armazenar a entrada como primeira ativação
        self.activations = [X]
        
        # Calcular ativação em cada camada
        for layer_model in self.layer_outputs:
            layer_output = layer_model.predict(X, verbose=0)
            self.activations.append(layer_output)
        
        # Retornar a saída final
        return self.activations[-1]
    
    def train(self, X, Y, epochs=100, learning_rate=0.01, verbose=True, 
             report_interval=100, target_loss=None, show_graph=False):
        """
        Treina a rede neural
        
        Args:
            X: Dados de entrada
            Y: Dados de saída desejados
            epochs: Número de épocas para treinar
            learning_rate: Taxa de aprendizado
            verbose: Se True, mostra informações durante o treinamento
            report_interval: Intervalo para reportar o progresso
            target_loss: Se especificado, para o treinamento quando a perda atingir este valor
            show_graph: Se True, mostra o gráfico de evolução durante o treinamento
            
        Returns:
            tuple: (histórico de perdas, gráfico de evolução)
        """
        # Atualizar a taxa de aprendizado do otimizador
        self.model.optimizer.learning_rate = learning_rate
        
        # Criar e iniciar o gráfico de evolução, se solicitado
        graph = None
        if show_graph:
            graph = EvolutionGraph(max_score=1.0)
            graph.run_in_thread()
        
        # Callback para monitorar perdas e atualizar o gráfico
        class CustomCallback(tf.keras.callbacks.Callback):
            def __init__(self, graph=None, target_loss=None):
                super().__init__()
                self.losses = []
                self.graph = graph
                self.target_loss = target_loss
                
            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get('loss')
                self.losses.append(loss)
                
                if self.graph:
                    self.graph.add_score(loss)
                
                if epoch % report_interval == 0 or epoch == epochs - 1:
                    print(f'Época {epoch}/{epochs}, Loss: {loss:.6f}')
                
                # Parada antecipada
                if self.target_loss and loss <= self.target_loss:
                    print(f"Parada antecipada na época {epoch}. Perda alvo de {self.target_loss:.6f} alcançada.")
                    self.model.stop_training = True
        
        # Criar o callback
        custom_callback = CustomCallback(graph=graph, target_loss=target_loss)
        
        # Treinar o modelo
        history = self.model.fit(
            X, Y,
            epochs=epochs,
            verbose=0,
            callbacks=[custom_callback]
        )
        
        # Manter o gráfico aberto por um momento para visualização
        if show_graph and graph:
            print("Gráfico de treinamento aberto. Fechará automaticamente em alguns segundos...")
            time.sleep(5)
        
        return custom_callback.losses, graph
    
    def predict(self, X):
        """
        Faz previsões para os dados de entrada
        
        Args:
            X: Dados de entrada
            
        Returns:
            np.array: Previsões
        """
        return self.model.predict(X, verbose=0)
    
    def get_weights_as_numpy(self):
        """
        Retorna os pesos do modelo como arrays numpy para compatibilidade com o visualizador
        
        Returns:
            tuple: (weights, biases)
        """
        weights = []
        biases = []
        
        for layer in self.model.layers:
            w, b = layer.get_weights()
            weights.append(w)
            biases.append(b)
        
        return weights, biases

def get_training_params():
    """Solicita parâmetros de treinamento do usuário"""
    try:
        epochs = int(input("Número de épocas (padrão=10000): ") or 10000)
        target_loss = float(input("Perda alvo para parada antecipada (padrão=0.001, 0 para desativar): ") or 0.001)
        learning_rate = float(input("Taxa de aprendizado (padrão=0.01): ") or 0.01)
        
        # Se target_loss for 0, desative a parada antecipada
        target_loss = target_loss if target_loss > 0 else None
        
        return epochs, target_loss, learning_rate
    except ValueError:
        print("Entrada inválida. Usando valores padrão.")
        return 10000, 0.001, 0.01

if __name__ == "__main__":
    # Dados fornecidos
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    t = np.array([-.9602, -0.5770, -0.0729, 0.3771, 0.6405, 0.6600, 0.4609, 0.1336, -0.2013, -0.4344, -0.5000])
    
    # Normalização dos dados de entrada e saída
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    t_min = np.min(t)
    t_max = np.max(t)
    t_norm = (t - t_min) / (t_max - t_min)
    
    # Preparando os dados para o MLP
    X_train = np.array([[val] for val in x_norm])
    Y_train = np.array([[val] for val in t_norm])
    
    # Pergunta ao usuário qual implementação usar
    print("\nEscolha a implementação a ser usada:")
    print("1 - NumPy (CPU)")
    if TF_AVAILABLE:
        print("2 - TensorFlow (GPU se disponível)")
    implementation = input("Opção (padrão=1): ") or "1"
    
    # Criar modelo conforme escolha do usuário
    if implementation == "2" and TF_AVAILABLE:
        print("\nUsando implementação TensorFlow")
        mlp = MLPTF(ninputs=1, nhidden=[12], noutputs=1)
    else:
        print("\nUsando implementação NumPy")
        mlp = MLP(ninputs=1, nhidden=[12], noutputs=1)
    
    # Obtendo parâmetros de treinamento do usuário
    print("\nConfiguração do treinamento do MLP:")
    epochs, target_loss, learning_rate = get_training_params()
    
    # Obter preferência do usuário para o gráfico de evolução
    print("\nDeseja visualizar o gráfico de evolução do treinamento em tempo real? (s/n)")
    show_graph = input().lower() in ['s', 'sim']
    
    # Treinando o modelo
    start_time = time.time()
    losses, graph = mlp.train(
        X_train, Y_train, 
        epochs=epochs, 
        learning_rate=learning_rate, 
        verbose=True, 
        report_interval=500, 
        target_loss=target_loss, 
        show_graph=show_graph
    )
    train_time = time.time() - start_time
    print(f"\nTempo de treinamento: {train_time:.2f} segundos")
    
    # Fazendo previsões
    predictions_norm = mlp.predict(X_train)
    
    # Formatando as previsões para corresponder à saída esperada
    if implementation == "2" and TF_AVAILABLE:
        predictions_norm = np.array([p[0] for p in predictions_norm])
    else:
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
    
    # Para a versão NumPy, mostramos os histogramas dos pesos
    if implementation == "1":
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.hist(mlp.weights[0].flatten(), bins=30)
        plt.title('Distribuição dos Pesos na Primeira Camada')
        
        plt.subplot(1, 2, 2)
        plt.hist(mlp.weights[-1].flatten(), bins=30)
        plt.title('Distribuição dos Pesos na Última Camada')
        
        plt.tight_layout()
    
    plt.show()
    
    # Se o gráfico foi criado, dê tempo para visualizá-lo antes de fechar
    if graph:
        print("Pressione Enter para continuar e fechar o gráfico de evolução...")
        input()
        graph.exit()
    
    # Adicionar visualização da rede neural
    print("\nDeseja visualizar a animação da rede neural? (s/n)")
    choice = input().lower()
    if choice == 's' or choice == 'sim':
        # Criar nova entrada para visualização com o formato correto
        test_point = 5  # Meio da escala original (0-10)
        test_norm = (test_point - np.min(x)) / (np.max(x) - np.min(x))
        test_input = np.array([test_norm])
        
        print("Executando visualização com entrada de teste...")
        
        # Para a versão TensorFlow, precisamos adaptar o visualizador
        if implementation == "2" and TF_AVAILABLE:
            # Limpar ativações anteriores
            mlp.activations = []
            
            # Obter ativações
            _ = mlp.forward(test_input)
            
            # Adaptar a interface do MLPVisualizer para usar o modelo TensorFlow
            class MLPAdapter:
                def __init__(self, tf_mlp):
                    self.tf_mlp = tf_mlp
                    self.weights, self.biases = tf_mlp.get_weights_as_numpy()
                    self.activations = [a if i == 0 else a.reshape(-1) 
                                       for i, a in enumerate(tf_mlp.activations)]
            
            # Criar adaptador
            mlp_adapter = MLPAdapter(mlp)
            
            # Visualizar a rede neural
            visualizer = MLPVisualizer(mlp_adapter)
            visualizer.animate()
        else:
            # Limpar ativações anteriores
            mlp.activations = []
            
            # Executar várias vezes para ter mais quadros na animação
            for _ in range(5):
                mlp.forward(test_input)
            
            # Visualizar a rede neural
            visualizer = MLPVisualizer(mlp)
            visualizer.animate()
