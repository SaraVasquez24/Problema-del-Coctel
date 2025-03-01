# Problema-del-Coctel
Realizado por Eduard Santiago Alarcon y Sara Damaris Vásquez Cardenas
### Problema del cóctel
# Análisis y Separación de Fuentes de Audio usando ICA y Beamforming

## Introducción

Este laboratorio se centra en la adquisición, procesamiento y análisis de señales de audio provenientes de diferentes micrófonos con el objetivo de separar fuentes de sonido utilizando Análisis de Componentes Independientes (ICA) y mejorar la direccionalidad mediante Beamforming. 

El sistema de obtención de audios considera la digitalización de los audios con criterios definidos y se realiza un análisis tanto en el dominio del tiempo como en el dominio de la frecuencia. Además, se calcula la Relación Señal-Ruido (SNR) para evaluar la calidad del resultado.

## Configuración del Sistema

El sistema de obtención de datos consiste en dos micrófonos colocados a una distancia de **5.12 metros** entre sí. Se capturan tres archivos de audio:
1. **voz_celular.wav** - Capturado con el primer micrófono.
2. **voz_ipad.wav** - Capturado con el segundo micrófono.
3. **silencio_ipad.wav** - Ruido capturado para el cálculo del SNR.

Los audios se digitalizan con una frecuencia de muestreo de **48 kHz**, asegurando un rango adecuado de captura de frecuencias. 

## Implementación del Código

El código realiza las siguientes operaciones:

### 1. Carga y Preprocesamiento de los Audios

Se cargan los archivos de audio y se convierten a formato mono si es necesario.
```python
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import resample, correlate
from scipy.fftpack import fft, fftfreq
from sklearn.decomposition import FastICA

# Cargar los audios mezclados
audio1, sr1 = sf.read("D:/LABORATORIOS_PDS/LAB_3/audios/voz_celular.wav")  # Primer micrófono
audio2, sr2 = sf.read("D:/LABORATORIOS_PDS/LAB_3/audios/voz_ipad.wav")  # Segundo micrófono
audio3, sr3 = sf.read("D:/LABORATORIOS_PDS/LAB_3/audios/silencio_ipad.wav")

# Convertir audios estéreo a mono (tomando solo un canal)
if len(audio1.shape) > 1:
    audio1 = audio1[:, 0]  # Tomamos el primer canal
if len(audio2.shape) > 1:
    audio2 = audio2[:, 0]  # Tomamos el primer canal
if len(audio3.shape) > 1:
    audio3 = audio3[:, 0]  # Tomamos el primer canal
```

### Asegurar que los audios tengan la misma longitud
```
min_len = min(len(audio1), len(audio2))
audio1, audio2 = audio1[:min_len], audio2[:min_len]
```
Se recorta el audio más largo para que ambos tengan la misma cantidad de muestras.

### 2. Separación de Fuentes con ICA

Se aplica **FastICA** para extraer componentes independientes de las grabaciones.
```python
# Aplicar ICA en cada audio para luego sumar
ica1 = FastICA(n_components=1)
fuente_separada1 = ica1.fit_transform(audio1.reshape(-1, 1)).flatten()

ica2 = FastICA(n_components=1)
fuente_separada2 = ica2.fit_transform(audio2.reshape(-1, 1)).flatten()
```
ICA es un método que trata de separar señales mezcladas en sus fuentes originales.
Cada audio se ajusta a la forma (N,1) para ser procesado por FastICA.
Se obtiene una señal separada para cada audio.
```
# Guardar audios 
sf.write("voz_separada1.wav", fuente_separada1, sr1)
sf.write("voz_separada2.wav", fuente_separada2, sr2)
```
Se guardan los audios después de aplicar ICA.


### 3. Suma de Voces Separadas

Se suman las fuentes separadas para analizar la combinación de las señales extraídas.
```python
suma_voces = fuente_separada1 + fuente_separada2
sf.write("voz_suma.wav", suma_voces, sr1)
```
### Análisis en el dominio del tiempo

```
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(audio1, label="Audio 1 Original")
plt.plot(audio2, label="Audio 2 Original", alpha=0.7)
plt.title("Señales Originales en el Dominio del Tiempo")
```
Se grafican las señales originales, separadas y sumadas en el dominio del tiempo.

### 4. Análisis en el Dominio del Tiempo y la Frecuencia

Se grafican las señales y sus respectivos espectros de frecuencia.
```python
def calcular_fft(senal, sr):
    N = len(senal)
    freqs = fftfreq(N, 1/sr)
    fft_values = np.abs(fft(senal))
    return freqs[:N // 2], fft_values[:N // 2]
```
`fft(senal):` Se calcula la Transformada Rápida de Fourier (FFT).
`fftfreq(N, 1/sr):` Se obtienen las frecuencias asociadas.

```
freq_audio1, fft_audio1 = calcular_fft(audio1, sr1)
freq_audio2, fft_audio2 = calcular_fft(audio2, sr2)
freq_fuente1, fft_fuente1 = calcular_fft(fuente_separada1, sr1)
freq_fuente2, fft_fuente2 = calcular_fft(fuente_separada2, sr2)
freq_suma, fft_suma = calcular_fft(suma_voces, sr1)
```
Se calcula la FFT para cada señal.

```
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(freq_audio1, fft_audio1, label="FFT Audio 1 Original")
plt.plot(freq_audio2, fft_audio2, label="FFT Audio 2 Original", alpha=0.7)
plt.title("Espectro de Frecuencia - Audios Originales")
```
Se grafican los espectros de frecuencia.


### 5. Beamforming

Se realiza beamforming estimando el retardo de tiempo entre los micrófonos para mejorar la direccionalidad de la captura de voz.
```python
vel_sonido = 343  # Velocidad del sonido en m/s
distancia_micros = 5.12  # Distancia entre micrófonos en metros
````
```
correlacion = correlate(audio1, audio2, mode='full')
lags = np.arange(-len(audio1) + 1, len(audio2))
```
Se calcula la correlación cruzada entre los dos audios para encontrar el desfase temporal.

```
max_offset = int(0.01 * sr1)
idx_limited = np.arange(len(correlacion) // 2 - max_offset, len(correlacion) // 2 + max_offset)
lag_optimo = lags[idx_limited][np.argmax(correlacion[idx_limited])]
tiempo_retardo = lag_optimo / sr1
```
Se limita la búsqueda de desplazamiento a `0.01` segundos (10 ms).
`lag_optimo` es el desplazamiento en muestras con la máxima correlación.
`tiempo_retardo` es la diferencia de tiempo entre los dos audios.
```
delta_d = tiempo_retardo * vel_sonido  
print(f"Retardo estimado: {tiempo_retardo:.6f} s ({lag_optimo} muestras)")
print(f"Diferencia de distancia recorrida: {delta_d:.3f} m")
```
```
if abs(lag_optimo) < len(audio1):  
    if lag_optimo > 0:
        audio2_aligned = np.pad(audio2, (lag_optimo, 0), mode='constant')[:len(audio1)]
    else:
        audio1_aligned = np.pad(audio1, (-lag_optimo, 0), mode='constant')[:len(audio2)]
```
Se alinea el audio retrasado aplicando `np.pad()`

```
audio_sum = (audio1_aligned + audio2_aligned) / 2
sf.write("voz_beamforming.wav", audio_sum, sr1)
```
Se promedia la señal alineada y se guarda.


### 6. Cálculo del SNR

Se calcula el **SNR** utilizando la señal de ruido capturada.
```python
def calcular_snr(senal, ruido):
    potencia_senal = np.mean(senal ** 2)
    potencia_ruido = np.mean(ruido ** 2)
    snr = 10 * np.log10(potencia_senal / potencia_ruido)
    return snr
```
Calcula la relación entre la potencia de la señal y el ruido en decibeles (dB).

```
snr_valor = calcular_snr(audio1, audio3)
print(f"El SNR es: {snr_valor:.2f} dB")
```
Se calcula el SNR usando `audio1` como señal y `audio3` como ruido.

### 7. Resultados

#### Analisis de la señal en el dominio de la frecuencia
![Imagen de WhatsApp 2025-02-28 a las 20 46 35_3bd4f477](https://github.com/user-attachments/assets/02f0b99e-6dd5-4ee6-afe6-14e322a23ca1)

![Imagen de WhatsApp 2025-02-28 a las 20 46 45_345d3e87](https://github.com/user-attachments/assets/02dabb7f-352d-489d-a385-7fc46c08362b)



## Bibliografía
-Cohen, L. (1995). Time-Frequency Analysis. Prentice Hall.
-Makino, S., & Sawada, H. (2007). Blind Signal Separation. Springer.
-(S/f). Mathworks.com. Recuperado el 1 de marzo de 2025, de https://la.mathworks.com/discovery/beamforming.html
-Google colab. (s/f). Google.com. Recuperado el 1 de marzo de 2025, de https://colab.research.google.com/github/fchirono/BeamformingBasics/blob/main/BeamformingBasics.ipynb
-Martínez, A., & Rodríguez, S. (2023). Eliminación de artefactos en el EEG basada en el análisis de componentes independientes. Universidad Complutense de Madrid. https://docta.ucm.es/entities/publication/fec541e4-64bc-4b48-bbdb-6a09e1c22155 

