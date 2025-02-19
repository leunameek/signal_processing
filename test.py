import matplotlib
matplotlib.use('TkAgg') 
import numpy as np
import matplotlib.pyplot as plt

#Emanuel Solarte Melo, Código 1202680

A = 10 #
omega = 24
t1 = -0.2
t2 = 0.3
t = np.arange(-0.5, 2, 0.0001)

def config_plot(signal, title):
    plt.figure()
    plt.plot(t, signal)
    plt.title(title)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.xlim(-0.5, 2)
    plt.show()
    
y = A * np.sin((omega * t) - t1) + np.cos((2 * omega * t) - t2)
config_plot(y, "Primer punto\nA = 10")