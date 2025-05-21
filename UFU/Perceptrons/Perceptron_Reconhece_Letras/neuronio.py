class Neuronio:
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

    def learnPerceptron(self, Data, weights=[], bias=-1, alpha=-1):
        self.n = nSignals = len(Data[0][0])

        #Step 0 (Initialize weights, bias and set learning rate)
        if not len(weights):
            weights =  [0 for _ in range(nSignals)]
        if bias == -1:
            bias = 0
        if alpha == -1:
            alpha = 1 # Learning Rate

        #Step 1 (While stop is false do steps 2-6)
        stop = False
        while not stop:
            stop = True
            #Step 2 (For each training pair s:t do steps 3-5)
            for i in range(len(Data)):
                d = Data[i]

                #Step 3 (Set ativation of inputs)
                x = d[0]
                t = d[1]

                #Step 4 (Compute response of output unit)
                XiWi =  [x[j] * weights[j] for j in range(nSignals)]
                yin = bias + sum(XiWi)
                l = self.l
                y = self.f(yin, l)

                #Step 5 (If an error: update weights and bias; Else: break step 2)
                if y != t:
                    newWeights = [weights[j]+ (alpha * t * x[j]) for j in range(nSignals)]
                    weights = newWeights[:]
                    bias = bias + alpha * t
                    stop = False

        self.w = weights
        self.b = bias
