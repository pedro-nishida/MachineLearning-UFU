# Objetivos
# - Treinar um Adaline usando a base de dados B2
# - Plotar o erro quadrático total durante o treinamento
# - Realizar o teste da rede neural treinada.

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from adaline import Adaline

# Extraindo os dados ================================

# Caminho para a base B2
base_path = "./Basedados_B2.xlsx"
# Dados da base B2
data = pd.read_excel(base_path)
adaline_data = [([l['s1'], l['s2']], int(l['t'])) for i,l in data.iterrows()]

# Adaline ===========================================
def ActivateFunction(Yin, Limiar):
    if Yin >= Limiar:
        return 1
    return -1

neuron = Adaline(ActivateFunction)
w, b, sqedT = neuron.learnAdaline(adaline_data, alpha=1e-6, tolerance=1e-12)

# Testando os resultados ============================
print(f"Pesos = {w}, bias = {b}")
for s,t in adaline_data:
    y = neuron.output(s)
    if(t==y):
        print(f"Certo: ({s=},{t=}): {y=}")
    else:
        print(f"Errado: ({s=},{t=}): {y=}")

# Plotando gráficos =================================
fig, ax = plt.subplots(1, 2)

# Valores de entrada
d1 = [d[0] for d in adaline_data if d[1]==1]
dm1 = [d[0] for d in adaline_data if d[1] == -1]
# Fronteira de separação
ax[0].set_title("Data e Fronteira de Separação")
ax[0].plot([_[0] for _ in d1], [_[1] for _ in d1], '.b', label="1")
ax[0].plot([_[0] for _ in dm1], [_[1] for _ in dm1], '.r', label="-1")
x = np.linspace(0, 3, 10000)
ax[0].plot(x, (lambda x,w,b : (-x*w[0]-b)/w[1])(x,w,b) , '-k', label='Fronteira')
ax[0].legend()
# Erro quadrático total
ax[1].set_title("Erro Quadrático Total")
ax[1].plot(sqedT, '-k')

plt.show()

