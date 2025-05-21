import numpy as np
import random
random.seed(12345)

class Adaline:
    def __init__(self, funcaoAtivacao, limiar=0, weights=[]):
        self.f = funcaoAtivacao
        self.l = limiar
        self.n = len(weights)
        self.w = weights
        self.b = 0

    def output(self, inputs):
        XiWi = [inputs[i] * self.w[i] for i in range(self.n)]
        y_in = self.b + sum( XiWi )
        y = self.f(y_in, self.l)
        return y

    def learnAdaline(self, Data, weights=np.array([]), bias=-1, alpha=-1, tolerance=1e-12):
        self.n = nSignals = len(Data[0][0])

        cycle = 1
        squareError = 0
        squareErrorData = []
        #Step 0 (Initialize weights and bias with small
            # random values [-0.5, 0.5] and set learning rate)
        if not len(weights):
            weights =  np.array([random.random()-0.5 for _ in range(nSignals)])
        if bias == -1:
            bias = random.random()-0.5
        if alpha == -1:
            alpha = 1 # Learning Rate

        #Step 1 (While stop is false do steps 2-6)
        stop = False
        while not stop:
            cycle+=1
            #Step 2 (For each training pair s:t do steps 3-5)
            for i in range(len(Data)):
                d = Data[i]
                #Step 3 (Set ativation of inputs)
                x = np.array(d[0])
                t = d[1]
                #Step 4 (Compute response of output unit)
                XiWi =  x * weights
                yin = bias + np.sum(XiWi)
                squareError += 0.5*(t-yin)**2
                #Step 5 Update Weights
                diffWeights = x * (alpha * (t - yin))
                weights = weights +  diffWeights
                bias += alpha * (t - yin)
                #Step 6 (If the largest weight change that occurred
                    # in Step 2 is lower than tolerance -> stop)
                if( np.max(np.abs(diffWeights)) < tolerance ):
                    stop = True
                    print(f"Ciclos: {cycle}")
            squareErrorData.append(squareError)

        self.w = weights
        self.b = bias
        return (weights, bias, squareErrorData)
