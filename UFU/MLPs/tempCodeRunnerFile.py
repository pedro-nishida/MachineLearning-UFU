import numpy as np
import numpy.random as npr

class MLP:
    def __init__(self, ninputs, nhidden, noutputs):
        # Número de camadas = número de camadas escondidas + camada de saída
        self.nlayers = len(nhidden) + 1

        # Inicialização dos pesos e vieses
        self.weights, self.biases = [], []

        # Random values
        self.weights.append(np.array([[npr.random()-0.5 for n in range(ninputs)] for Layer_0 in range(nhidden[0])]))
        self.biases.append(np.array([npr.random()-0.5 for Layer_0 in range(nhidden[0])]))

        for i in range(1, len(nhidden)):
            self.weights.append(np.array([[npr.random()-0.5 for n in range(nhidden[i-1])] for Layer_i in range(nhidden[i])]))
            self.biases.append(np.array([npr.random()-0.5 for Layer_i in range(nhidden[i])]))

        self.weights.append(np.array([[npr.random()-0.5 for n in range(nhidden[-1])] for Layer_Out in range(noutputs)]))
        self.biases.append(np.array([npr.random()-0.5 for Layer_out in range(noutputs)]))

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

        deltas=[]
        # Retropropagação do erro
        deltaK = (self.activations[-1] - y)*dx(self.y[-1])
        for i in range(len(self.weights[-1])):
            for j in range(len(self.weights[-1][i])):
                self.weights[-1][i][j] += learning_rate*deltaK[i]*self.y[-2][j]
            self.biases[-1][i]+=learning_rate*deltaK[i]

        deltas.append(deltaK)
        for i in range(len(self.weights) - 2, 0, -1):
            for j in range(len(self.weights[i])):
                deltainj = np.sum(deltaK[-1]* np.sum(self.weights[i+1][j]))
                deltaj = deltainj*dx(self.y[i])
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j] += learning_rate*deltaj*self.y[i]

                # print(np.shape(self.biases[i]),np.shape(deltaj))
                self.biases[i][j]+=learning_rate*deltaj

        #     # Atualização dos pesos e vieses
        #     self.weights[i] -= learning_rate * dw
        #     self.biases[i] -= learning_rate * db

        # dw = np.dot(X.T, dz)
        # db = np.sum(dz, axis=0, keepdims=True)

        # self.weights[0] -= learning_rate * dw
        # self.biases[0] -= learning_rate * db



    def train(self, X, Y, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            print(epoch)
            for i in range(np.shape(X)[0]):
                x = X[i]
                y = Y[i]

                print(f"Epóca {epoch} ->   Fowarding", end='\r')
                output = self.forward(x)

                print(f"Epóca {epoch} -> Backwarding", end='\r')
                self.backward(x, y, learning_rate)

                if epoch % 10 == 0:
                    loss = -np.sum(y * np.log(output)) / len(y)
                    print(f'Epoch {epoch}, Loss: {loss}')
