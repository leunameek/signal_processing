import matplotlib
matplotlib.use('TkAgg') 
import numpy as np
import matplotlib.pyplot as plt

# Solicitar valores al usuario
print("Hola, escribe los valores para tu señal inicial discreta!\n")
A = float(input("Ingrese el valor de A: "))
omega = float(input("Ingrese el valor de w: "))
n0 = float(input("Ingrese el valor de n0: "))

# Configurar el tiempo de 0 a 5 segundos
t = np.linspace(0, 5, 50)

def config_stem_plot(signal, title):
    plt.figure()
    plt.stem(t, signal)
    plt.title(title)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.xlim(0, 5)
    plt.show()

# Señal original y(t) = A * sen(w*t - n0)
y = A * np.sin(omega * t - n0)
config_stem_plot(y, f"Parte 1: Señal original\nA = {A}, w = {omega}, n0 = {n0}")

# Multiplicar por escalón unitario con retraso de 0.5s
u = np.where(t >= 0.5, 1.0, 0.0)  # Escalón unitario retardado
y2 = y * u
config_stem_plot(y2, f"Parte 2: Señal x escalón en t=0.5s \nA = {A}, w = {omega}, n0 = {n0}")

# Sumar rampa descendente con adelanto de 0.25s
ramp = - (t + 0.25)  # Rampa descendente y adelantada
y3 = y2 + ramp
config_stem_plot(y3, f"Parte 3: Señal + rampa\nA = {A}, w = {omega}, n0 = {n0}")