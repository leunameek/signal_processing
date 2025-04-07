import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.io import loadmat
from scipy.fft import fft, fftfreq

# Definimos la variable
x = sp.Symbol('x')

# Función original: f(x) = x / pi en [-pi, pi]
f = x / sp.pi

# Calculamos los coeficientes de Fourier (hasta n=3)
a0 = (1 / (2 * sp.pi)) * sp.integrate(f, (x, -sp.pi, sp.pi))
b1 = (1 / sp.pi) * sp.integrate(f * sp.sin(x), (x, -sp.pi, sp.pi))
b2 = (1 / sp.pi) * sp.integrate(f * sp.sin(2 * x), (x, -sp.pi, sp.pi))
b3 = (1 / sp.pi) * sp.integrate(f * sp.sin(3 * x), (x, -sp.pi, sp.pi))

print(f'a0 = {a0.evalf()}')
print(f'b1 = {b1.evalf()}')
print(f'b2 = {b2.evalf()}')
print(f'b3 = {b3.evalf()}')

# Convertimos los coeficientes a float
a0 = float(a0)
b1 = float(b1)
b2 = float(b2)
b3 = float(b3)

# Función original (numérica)
f_original = lambda x: x / np.pi

# Aproximación con Serie de Fourier
def fourier_series(x):
    return a0/2 + b1 * np.sin(x) + b2 * np.sin(2*x) + b3 * np.sin(3*x)

# Valores para graficar
x_vals = np.linspace(-np.pi, np.pi, 1000)
f_vals = f_original(x_vals)
fourier_vals = fourier_series(x_vals)

# Cargar archivo .mat
mat_data = loadmat('signal.mat')
print(mat_data.keys())  # Verifica cómo se llama la señal

signal = mat_data['signal'].squeeze()

fs = 256  # frecuencia en hertzzzzz
t = np.arange(len(signal)) / fs

N = len(signal)
yf = fft(signal)
xf = fftfreq(N, 1/fs)

# Filtramos solo la parte positiva
idx = xf >= 0
xf = xf[idx]
yf = np.abs(yf[idx])

plt.figure(figsize=(12, 12))

# Función original hecha en clase
plt.subplot(3,1,1)
plt.plot(x_vals, f_vals, 'b', linewidth=2, label='Función original')
plt.plot(x_vals, fourier_vals, 'r--', linewidth=2, label='Serie de Fourier (n=3)')
plt.legend()
plt.title('Función f(x) = x/π y su Serie de Fourier')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)

# Señal del profe
plt.subplot(3,1,2)
plt.plot(t, signal, color='purple')
plt.title('Señal EEG desde signal.mat')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)

# FFT de la señal del profe
plt.subplot(3,1,3)
plt.plot(xf, yf, color='green')
plt.title('FFT de la señal EEG')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud')
plt.grid(True)

plt.tight_layout()
plt.show()
