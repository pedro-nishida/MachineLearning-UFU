import numpy as np
import random

def f(x):
    return x

class Adaline:
    def __init__(self):
        self.w = 0
        self.b = 0

    def output(self, inputs):
        XiWi = inputs * self.w
        y = self.b + XiWi
        return y

    def learnAdaline(self, Data, cycles = 1000, weights=False, bias=False, alpha=False, tolerance=1e-12):
        self.n = 1

        cycle = 1
        squareError = 0
        squareErrorData = []
        #Step 0 (Initialize weights and bias with small
            # random values [-0.5, 0.5] and set learning rate)
        if not weights:
            weights = random.random()-0.5
        if not bias:
            bias = random.random()-0.5
        if not alpha:
            alpha = 1 # Learning Rate

        #Step 1 (While stop is false do steps 2-6)
        stop = False
        while not stop:
            if cycle == cycles:
                print(f"Ciclos: {cycle}")
                break
            cycle+=1
            # print(f"{cycle=}: {weights=}, {bias=}")
            #Step 2 (For each training pair s:t do steps 3-5)
            for i in range(len(Data)):
                d = Data[i]
                #Step 3 (Set ativation of inputs)
                x = d[0]
                t = d[1]
                #Step 4 (Compute response of output unit)
                yin =  (x * weights) + bias
                squareError += 0.5*(t-yin)*(t-yin)
                #Step 5 Update Weights
                diffWeights = x * alpha * (t - yin)
                weights +=  diffWeights
                bias += alpha * (t - yin)
                #Step 6 (If the largest weight change that occurred
                    # in Step 2 is lower than tolerance -> stop)
                if( abs(diffWeights) < tolerance ):
                    stop = True
                    print(f"Ciclos: {cycle}")
            squareErrorData.append(squareError)

        self.w = weights
        self.b = bias
        return (weights, bias, squareErrorData)
