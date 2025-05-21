def ActivateFunction(Yin, Limiar):
    if Yin >= Limiar: return 1
    return -1

def PerceptronLearn(Data, Limiar = 0, weights=[], bias=-1, alpha=-1):
    nSignals = len(Data[0][0])

    #Step 0 (Initialize weights, bias and set learning rate)
    if not len(weights):
        weights = [0] * nSignals
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
            y_in = bias + sum(XiWi)
            y = ActivateFunction(y_in, Limiar)

            #Step 5 (If an error: update weights and bias; Else: break step 2)
            if y != t:
                newWeights = [weights[j]+ (alpha * t * x[j]) for j in range(nSignals)]
                weights = newWeights[:]
                bias = bias + alpha * t
                stop = False

    return (weights,bias)

# Testes
def Teste():
    DataE =(([ 1, 1],  1),
            ([ 1,-1], -1),
            ([-1, 1], -1),
            ([-1,-1], -1))

    DataOU=(([ 1, 1],  1),
            ([ 1,-1],  1),
            ([-1, 1],  1),
            ([-1,-1], -1))

    w, b = PerceptronLearn(DataE)
    print(f"E: {w=} {b=}")
    for d in DataE:
        x = d[0]
        y_in = b + x[0]*w[0] + x[1]*w[1]
        y = ActivateFunction(y_in, 0)
        if y != d[1]:
            print(f"Problema no Perceptron- {w=} {b=}\n{x=}: Esperado {d[1]} Encontrado {y}")

    w, b = PerceptronLearn(DataOU)
    for d in DataOU:
        x = d[0]
        y_in = b + x[0]*w[0] + x[1]*w[1]
        y = ActivateFunction(y_in, 0)
        if y != d[1]:
            print(f"Problema no Perceptron- {w=} {b=}\n{x=}: Esperado {d[1]} Encontrado {y}")
    print(f"OU: {w=} {b=}")

Teste()