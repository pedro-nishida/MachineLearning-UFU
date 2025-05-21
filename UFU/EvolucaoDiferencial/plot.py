import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Função de Rastrigin (2D para exemplo)
def rastrigin(x, y):
    return 10*2 + (x**2 + y**2) - 10*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))

# Grid para plotar a função
x = np.linspace(-5.12, 5.12, 100)
y = np.linspace(-5.12, 5.12, 100)
X, Y = np.meshgrid(x, y)
Z = rastrigin(X, Y)

# Plot da função
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

# Adicione seus resultados (ex.: melhores pontos por geração)
resultados_x = np.array([...])  # Substitua pelos seus dados
resultados_y = np.array([...])
resultados_z = rastrigin(resultados_x, resultados_y)
ax.scatter(resultados_x, resultados_y, resultados_z, color='red', s=50, label='Evolução Diferencial')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.legend()
plt.show()