import matplotlib
matplotlib.use('TkAgg') 
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
plt.subplots_adjust(hspace=0.6, top=0.95, bottom=0.05)

# Parámetros comunes
t = np.arange(-0.02, 0.02, 0.0001)

# Función para configurar subplots consistentemente
def config_subplot(ax, title):
    ax.set_title(title, fontsize=10, pad=12)
    ax.grid(True)
    ax.set_xlim(-0.02, 0.02)
    ax.tick_params(axis='both', which='major', labelsize=8)

# 1. Señal senoidal (Señal A)
ax1 = plt.subplot(6, 1, 1)
A = 4 * np.sin(2 * np.pi * t / 0.01)
ax1.plot(t, A)
config_subplot(ax1, 'Señal A: Seno de 10ms 4 unidades de amplitud')

# 2. Escalón unitario (Señal B)
ax2 = plt.subplot(6, 1, 2)
B = 1.0 * (t >= 0)
ax2.plot(t, B)
config_subplot(ax2, 'Señal B: Escalón unitario')

# 3. Suma A + B
ax3 = plt.subplot(6, 1, 3)
ax3.plot(t, A + B)
config_subplot(ax3, 'Suma A + B')

# 4. Multiplicación A * B
ax4 = plt.subplot(6, 1, 4)
ax4.plot(t, A * B)
config_subplot(ax4, 'Multiplicación A * B')

# 5. Corrimiento de A en atraso (3ms)
ax5 = plt.subplot(6, 1, 5)
A_shift = 4 * np.sin(2 * np.pi * (t - 0.003) / 0.01)
ax5.plot(t, A_shift)
config_subplot(ax5, 'Señal A con retardo de 3ms')

# 6. Corrimiento de B en adelanto (5ms)
ax6 = plt.subplot(6, 1, 6)
B_shift = 1.0 * (t >= -0.005)
ax6.plot(t, B_shift)
config_subplot(ax6, 'Señal B con adelanto de 5ms')

# Ajuste automático de layout
plt.tight_layout()
plt.show()