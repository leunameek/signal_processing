import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Definimos la variable
x = sp.Symbol('x')

# Definimos la función original
f = x / sp.pi

# Calcula los coeficientes de Fourier
a0 = (1 / (2 * sp.pi)) * sp.integrate(f, (x, -sp.pi, sp.pi))
b1 = (1 / sp.pi) * sp.integrate(f * sp.sin(x), (x, -sp.pi, sp.pi))
b2 = (1 / sp.pi) * sp.integrate(f * sp.sin(2 * x), (x, -sp.pi, sp.pi))
b3 = (1 / sp.pi) * sp.integrate(f * sp.sin(3 * x), (x, -sp.pi, sp.pi))

print(f'a0 = {a0.evalf()}')
print(f'b1 = {b1.evalf()}')
print(f'b2 = {b2.evalf()}')
print(f'b3 = {b3.evalf()}')

# Convierte a float
a0 = float(a0)
b1 = float(b1)
b2 = float(b2)
b3 = float(b3)

# Función original (numerica)
f_original = lambda x: x / np.pi

# Serie de Fourier
def fourier_series(x):
    return a0/2 + b1 * np.sin(x) + b2 * np.sin(2*x) + b3 * np.sin(3*x)

# Valores para graficar
x_vals = np.linspace(-np.pi, np.pi, 1000)
f_vals = f_original(x_vals)
fourier_vals = fourier_series(x_vals)

plt.figure(figsize=(8, 10))

plt.subplot(2,1,1)
plt.plot(x_vals, f_vals, 'b', linewidth=2, label='Función original')
plt.legend()
plt.title('Función original')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(x_vals, f_vals, 'b', linewidth=2, label='Función original')
plt.plot(x_vals, fourier_vals, 'r--', linewidth=2, label='Serie de Fourier')
plt.legend()
plt.title('Función original y su serie de Fourier')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)

plt.tight_layout()
plt.show()
