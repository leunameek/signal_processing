import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

#Serie Discreta de fourier
def periodic_signal(f, N, T):
    t = np.linspace(0, T, N, endpoint=False)
    x = f(t)  # función programable
    return t, x

def discrete_fourier(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-1j * 2 * np.pi * k * n / N)
    return X

def compute_audio_fft(audio_path):
    fs, audio = wavfile.read(audio_path)
    
    #mi audio está en stereo entonces lo paso a mono con esta bloquecito de código
    if len(audio.shape) == 2:
        audio = audio.mean(axis=1)

    N = len(audio)
    frequencies = np.fft.fftfreq(N, d=1/fs)
    X_fft = np.fft.fft(audio)

    half = N // 2
    frequencies = frequencies[:half]
    X_fft = X_fft[:half]

    return frequencies, X_fft

def freq_analysis(frequencies, X_fft, top=10):
    magnitudes = np.abs(X_fft)
    org_indexes = np.argsort(magnitudes)[::-1]

    print("\nFrecuencias predominantes:")
    for i in range(top):
        print(f"Frecuencia: {frequencies[org_indexes[i]]:.2f} Hz, Amplitud: {magnitudes[org_indexes[i]]:.2f}")

    print("\nFrecuencias menos importantes:")
    for i in range(-1, -top-1, -1):
        print(f"Frecuencia: {frequencies[org_indexes[i]]:.2f} Hz, Amplitud: {magnitudes[org_indexes[i]]:.2f}")

if __name__ == "__main__":

    print("\n Serie discreta de Fourier (Programable)\n")
    fs = 100  # muestras
    T = 1.4     # periodo en segundos
    f0 = 34    # frecuencia de la señal
    
    f = lambda t: np.sin(2*np.pi*f0*t)
    
    t, x = periodic_signal(f, fs, T)
    X = discrete_fourier(x)
    
    plt.figure()
    plt.stem(np.abs(X))
    plt.title("Serie Discreta de Fourier (DFS)")
    plt.xlabel("k")
    plt.ylabel("|X[k]|")
    plt.grid()
    plt.show()
    print("\n FFT de señal de audio \n")
    
    audio_path = 'ES_Wam Energy - HATAMITSUNAMI.wav'

    frequencies, X_fft = compute_audio_fft(audio_path)

    plt.figure()
    plt.plot(frequencies, np.abs(X_fft))
    plt.title("FFT de la señal de audio")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("|X[k]|")
    plt.grid()
    plt.show()
    
    print("\n Análisis de frecuencias predominantes \n")
    freq_analysis(frequencies, X_fft, top=10)
