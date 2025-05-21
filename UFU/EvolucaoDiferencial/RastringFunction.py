"""
Optimization of the Rastrigin test function
===========================================
"""

# %%
# The Rastrigin function is defined by:
#
# .. math::
#
#   f(\vect{x}) = A + \sum_{i=1}^n \left[x_i^2 - A\cos(2 \pi x_i)\right]
#
# where :math:`A=10` and :math:`\vect{x}\in[-5.12,5.12]^n`.
#
# It has a global minimum at :math:`\vect{x} = \vect{0}` where :math:`f(\vect{x})= - 10`.
#
# This function has many local minima, so optimization algorithms must be run from multiple starting points.
#
# In our example, we consider the bidimensional case, i.e. :math:`n=2`.
#
# **References**:
#
# - Rastrigin, L. A. "Systems of extremal control." Mir, Moscow (1974).
# - Rudolph, G. "Globale Optimierung mit parallelen Evolutionsstrategien". Diplomarbeit. Department of Computer Science, University of Dortmund, July 1990.
#

# %%
# Definition of the problem
# -------------------------

# %%
import openturns as ot
import openturns.viewer as otv
import math as m

ot.Log.Show(ot.Log.NONE)


def rastriginPy(X):
    A = 10.0
    delta = [x**2 - A * m.cos(2 * m.pi * x) for x in X]
    y = A + sum(delta)
    return [y]


dim = 2
rastrigin = ot.PythonFunction(dim, 1, rastriginPy)
print(rastrigin([1.0, 1.0]))

# %%
# Making `rastrigin` into a :class:`~openturns.MemoizeFunction` will make it recall all evaluated points.

# %%
rastrigin = ot.MemoizeFunction(rastrigin)

# %%
# This example is academic and the point achieving the global minimum of the function is known.

# %%
xexact = [0.0] * dim
print(xexact)

# %%
# The optimization bounds must be specified.

# %%
lowerbound = [-4.4] * dim
upperbound = [5.12] * dim
bounds = ot.Interval(lowerbound, upperbound)

# %%
# Plot the iso-values of the objective function
# ---------------------------------------------

# %%
graph = rastrigin.draw(lowerbound, upperbound, [100] * dim)
graph.setTitle("Rastrigin function")
graph.setLegendPosition("upper left")
graph.setLegendCorner([1.0, 1.0])
view = otv.View(graph)

# %%
# We see that the Rastrigin function has several local minima. However, there is only one single global minimum at :math:`\vect{x}^\star=(0, 0)`.

# %%
# Create the problem and set the optimization algorithm
# -----------------------------------------------------

# %%
problem = ot.OptimizationProblem(rastrigin)

# %%
# We use the :class:`~openturns.Cobyla` algorithm and run it from multiple starting points selected by a :class:`~openturns.LowDiscrepancyExperiment`.

# %%
size = 64
distribution = ot.JointDistribution([ot.Uniform(lowerbound[0], upperbound[0])] * dim)
experiment = ot.LowDiscrepancyExperiment(ot.SobolSequence(), distribution, size)
solver = ot.MultiStart(ot.Cobyla(problem), experiment.generate())

# %%
# Visualize the starting points of the optimization algorithm
# -----------------------------------------------------------

# %%
startingPoints = solver.getStartingSample()
graph = rastrigin.draw(lowerbound, upperbound, [100] * dim)
graph.setTitle("Rastrigin function")
cloud = ot.Cloud(startingPoints)
cloud.setPointStyle("bullet")
cloud.setColor("black")
graph.add(cloud)
graph.setLegends([""])
# sphinx_gallery_thumbnail_number = 2
view = otv.View(graph)

# %%
# We see that the starting points are well spread across the input domain of the function.

# %%
# Solve the optimization problem
# ------------------------------

# %%
solver.run()
result = solver.getResult()
xoptim = result.getOptimalPoint()
print(xoptim)

# %%
xexact

# %%
# We can see that the solver found a very accurate approximation of the exact solution.

# %%
# Analyze the optimization process
# --------------------------------
#
# :class:`~openturns.MultiStart` ran an instance of :class:`~openturns.Cobyla` from each starting point.
#
# Let us focus on the instance that found the global minimum. How many times did it evaluate `rastrigin`?

# %%
result.getCallsNumber()

# %%
# Let us view these evaluation points.

# %%
inputSample = result.getInputSample()
graph = rastrigin.draw(lowerbound, upperbound, [100] * dim)
graph.setTitle("Rastrigin function")
cloud = ot.Cloud(inputSample)
cloud.setPointStyle("bullet")
cloud.setColor("black")
graph.add(cloud)
graph.setLegendCorner([1.0, 1.0])
graph.setLegendPosition("upper left")
view = otv.View(graph)

# %%
# How fast did it find the global minimum?

# %%
graph = result.drawOptimalValueHistory()
view = otv.View(graph)

# %%
# Let us now analyze the :class:`~openturns.MultiStart` process as a whole.
#
# Since `rastrigin` is a :class:`~openturns.MemoizeFunction`,
# it has a :meth:`~openturns.MemoizeFunction.getInputHistory` method
# which lets us see all points it was evaluated on since its creation.

# %%
inputSample = rastrigin.getInputHistory()
graph = rastrigin.draw(lowerbound, upperbound, [100] * dim)
graph.setTitle("Rastrigin function")
cloud = ot.Cloud(inputSample)
cloud.setPointStyle("bullet")
cloud.setColor("black")
graph.add(cloud)
graph.setLegendCorner([1.0, 1.0])
graph.setLegendPosition("upper left")
view = otv.View(graph)

# %%
# How many times did all :class:`~openturns.Cobyla` instances combined call `rastrigin`?
print(rastrigin.getInputHistory().getSize())

# %%
otv.View.ShowAll()

# %%
# Add a 3D visualization of the Rastrigin function
# -----------------------------------------------

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a mesh grid for the 3D plot
x = np.linspace(lowerbound[0], upperbound[0], 100)
y = np.linspace(lowerbound[1], upperbound[1], 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros(X.shape)

# Evaluate the Rastrigin function on the mesh grid
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = rastriginPy([X[i, j], Y[i, j]])[0]

# Create the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X, Y)')
ax.set_title('Rastrigin Function - 3D View')

# Add the optimal point to the 3D plot
ax.scatter(xoptim[0], xoptim[1], rastriginPy(xoptim)[0], color='red', s=100, label='Optimal point')
ax.legend()

plt.tight_layout()
plt.show()

# %%
# We can also add the starting points to the 3D visualization
# ----------------------------------------------------------

# %%
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

# Add starting points to the 3D plot
for point in startingPoints:
    ax.scatter(point[0], point[1], rastriginPy(point)[0], color='black', s=20)

# Add the optimal point to the 3D plot
ax.scatter(xoptim[0], xoptim[1], rastriginPy(xoptim)[0], color='red', s=100, label='Optimal point')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X, Y)')
ax.set_title('Rastrigin Function with Starting Points and Optimal Solution')
ax.legend()

plt.tight_layout()
plt.show()
