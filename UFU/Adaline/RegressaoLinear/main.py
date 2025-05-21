from adaline import Adaline
from matplotlib import pyplot as plt
import numpy as np

Data = []
with open("C:/Users/phfuj/OneDrive/Documentos/Programação/AmaqRef/ML-Reference/Adaline/RegressaoLinear/basedeobservacoes.txt", "r") as file:
    for l in file:
        l = l.strip().split(" ")
        Data.append( (l[0], l[-1]) )
Data = [ ( float(d[0]), float(d[1]) ) for d in Data[1:]]

neuronio = Adaline()
w, b, sqedT = neuronio.learnAdaline(Data, 10000, alpha=1e-3, tolerance=1e-9)

for d in Data:
    k = neuronio.output(d[0])
    print(f"y={d[1]}   \t\tyAda={k}   \t\tErro={k-d[1]}")

# Coeficientes ======================================
x = np.array([d[0] for d in Data])
y = np.array([d[1] for d in Data])
yada = neuronio.output(x)

n=len(y)
sy = np.sum(y)
sy2 = np.sum(y**2)
syada = np.sum(yada)
syada2 = np.sum(yada**2)
syyada = np.sum(y*yada)
# Pearson
r = ( n*syyada-(sy*syada) ) / (((n*sy2) - (sy**2))**0.5 * ((n*syada2) - (syada**2))**0.5)
# Determinação
r2 = r**2


# Plotando gráficos =================================
fig, ax = plt.subplots(1, 1)

ax.set_title(f"y = ({w:.5f})x + ({b:.5f})\nPearson={r:.5f}  Determinação={r2:.5f}")
ax.plot(x,y, '.b', label="y")
ax.plot(x, yada, '.r', label="Adaline(x)")
# Regressão Linear
xl = np.linspace(-4, 11, 1000000)
ax.plot(xl, neuronio.output(xl), '-k', label="Regressão Linear")

fig.savefig("fig1.png")
plt.show()