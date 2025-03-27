import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

def x_lambda(l):
    return np.where((l >= 0) & (l < 1), 1, np.where((l >= 1) & (l < 2), -1, 0))

def v_lambda(l):
    return np.where((-1 <= l) & (l < 0), 1, 0)

lambda_range = np.arange(-3, 5, 0.1)
x_vals = x_lambda(lambda_range)
v_vals = v_lambda(lambda_range)

sampling_step = np.diff(lambda_range)[0]  # Hay que generar un intervalo de muestra para mostrar correctamente.
# Convolve es la función de la librería scipy para realizar convoluciones.
y_function = convolve(x_vals, v_vals, mode='full') * sampling_step

def convolution_algorithm(x, v):
    N = len(x)
    M = len(v)
    y = np.zeros(N + M - 1)
    
    for i in range(N + M - 1):
        for j in range(M):
            if (i - j) >= 0 and (i - j) < N:
                y[i] += x[i - j] * v[j]
    
    return y * sampling_step

y_manual = convolution_algorithm(x_vals, v_vals)

# Aqui calculamos el rango de nuestro resultado, de lo contrario no se verá en su totalidad y no será posible mostrarlo.
conv_time_range = np.arange(lambda_range[0] + lambda_range[0], 
                            lambda_range[0] + lambda_range[0] + sampling_step * (len(y_manual)), 
                            sampling_step)

plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.stem(lambda_range, x_vals, linefmt='b-', markerfmt='bo', basefmt='r-', label='x(λ)')
plt.title('Función x(λ)')
plt.grid()
plt.legend()

plt.subplot(4, 1, 2)
plt.stem(lambda_range, v_vals, linefmt='r-', markerfmt='ro', basefmt='r-', label='v(λ)')
plt.title('Función v(λ)')
plt.grid()
plt.legend()

plt.subplot(4, 1, 3)
plt.stem(conv_time_range, y_manual, linefmt='g-', markerfmt='go', basefmt='r-', label='Convolución Manual')
plt.title('Convolución de x(λ) y v(λ)')
plt.grid()
plt.legend()

plt.subplot(4, 1, 4)
plt.stem(conv_time_range, y_function, linefmt='m--', markerfmt='mo', basefmt='r-', label='Convolución Función')
plt.title('Convolución de x(λ) y v(λ)')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
