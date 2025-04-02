import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def periodic_signal(f, N, T):
    t = np.linspace(0, T, N, endpoint=False)
    x = f(t)
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
    
    if len(audio.shape) == 2:  # Pasamos a mono pq mi canción está en stereo
        audio = audio.mean(axis=1)

    N = len(audio)
    t_audio = np.linspace(0, N/fs, N, endpoint=False)
    
    frequencies = np.fft.fftfreq(N, d=1/fs)
    X_fft = np.fft.fft(audio)

    half = N // 2
    frequencies = frequencies[:half]
    X_fft = X_fft[:half]

    return t_audio, audio, frequencies, X_fft

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
  
    print("\nSerie discreta de Fourier\n")
    fs = 150
    T = 1.4
    f0 = 34

    f = lambda t: np.sin(2*np.pi*f0*t)

    t, x = periodic_signal(f, fs, T)
    X = discrete_fourier(x)
    
    plt.figure(figsize=(10,5))
    
    plt.subplot(1,2,1)
    plt.stem(t, x)
    plt.title("Señal programable en el dominio temporal")
    plt.xlabel("t [s]")
    plt.ylabel("x(t)")
    plt.grid()

    plt.subplot(1,2,2)
    plt.stem(np.abs(X))
    plt.title("Serie Discreta de Fourier (DFS)")
    plt.xlabel("k")
    plt.ylabel("|X[k]|")
    plt.grid()

    plt.tight_layout()
    plt.show()

    print("\nFFT de señal de audio\n")
    audio_path = 'ES_Wam Energy - HATAMITSUNAMI.wav'

    t_audio, audio, frequencies, X_fft = compute_audio_fft(audio_path)

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.plot(t_audio, audio)
    plt.title("Señal de audio en el dominio temporal")
    plt.xlabel("t [s]")
    plt.ylabel("Amplitud")
    plt.grid()

    plt.subplot(1,2,2)
    plt.plot(frequencies, np.abs(X_fft))
    plt.title("FFT de la señal de audio")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("|X[k]|")
    plt.grid()

    plt.tight_layout()
    plt.show()

    print("\nAnálisis de frecuencias\n")
    freq_analysis(frequencies, X_fft, top=10)
