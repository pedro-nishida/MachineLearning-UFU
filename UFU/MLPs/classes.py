import numpy as np

class MLP:
    def __init__(self, ninputs, nhidden, noutputs):
        """
        Inicializa uma rede neural MLP
        ninputs: número de entradas
        nhidden: lista de neurônios em cada camada oculta
        noutputs: número de saídas
        """
        # Estrutura da rede
        self.layers = [ninputs] + nhidden + [noutputs]
        self.n_layers = len(self.layers)
        
        # Inicializar pesos aleatoriamente
        self.weights = []
        self.biases = []
        
        for i in range(1, self.n_layers):
            # Pesos entre camadas adjacentes
            w = np.random.randn(self.layers[i], self.layers[i-1]) * 0.1
            self.weights.append(w)
            
            # Bias para cada neurônio da camada
            b = np.random.randn(self.layers[i], 1) * 0.1
            self.biases.append(b)
            
        # Para armazenar os valores durante o forward pass
        self.activations = []
        self.z_values = []
        
    def sigmoid(self, x):
        """Função de ativação sigmoid"""
        return 1.0 / (1.0 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivada da função sigmoid"""
        return self.sigmoid(x) * (1 - self.sigmoid(x))
        
    def forward(self, input_data):
        """
        Realiza a propagação para frente através da rede
        input_data: array de forma (batch_size, ninputs)
        """
        # Resetar as ativações
        self.activations = []
        self.z_values = []
        
        # Garantir formato correto da entrada
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        
        # A entrada é a primeira ativação
        self.activations.append(input_data.T)  # Transpor para (ninputs, batch_size)
        
        # Para cada camada (exceto a de entrada)
        for i in range(self.n_layers - 1):
            # z = w*a + b
            z = np.dot(self.weights[i], self.activations[i]) + self.biases[i]
            self.z_values.append(z)
            
            # a = sigmoid(z)
            a = self.sigmoid(z)
            self.activations.append(a)
            
        # Retornar a saída transposada para (batch_size, noutputs)
        return self.activations[-1].T
    
    def backward(self, x, y, learning_rate=0.1):
        """
        Realiza a retropropagação do erro e atualiza os pesos
        x: entrada, formato (batch_size, ninputs)
        y: saída desejada, formato (batch_size, noutputs)
        learning_rate: taxa de aprendizado
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        if len(y.shape) == 1:
            y = y.reshape(1, -1)
            
        y = y.T  # Transpor para (noutputs, batch_size)
        batch_size = x.shape[0]
        
        # Calcular o erro na camada de saída
        delta = (self.activations[-1] - y) * self.sigmoid_derivative(self.z_values[-1])
        
        # Retropropagar o erro
        for i in range(self.n_layers - 2, -1, -1):
            # Atualizar pesos e bias
            self.weights[i] -= learning_rate * np.dot(delta, self.activations[i].T) / batch_size
            self.biases[i] -= learning_rate * np.sum(delta, axis=1, keepdims=True) / batch_size
            
            # Calcular delta para a próxima iteração (camada anterior)
            if i > 0:  # Não precisamos calcular para a camada de entrada
                delta = np.dot(self.weights[i].T, delta) * self.sigmoid_derivative(self.z_values[i-1])