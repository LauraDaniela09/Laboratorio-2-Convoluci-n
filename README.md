# Laboratorio-2-Convolución-y-correlación
Reconocer la importancia de la aplicación de herramientas  matemáticas como la convolución y correlación en el área de procesamiento de  señales. 


## Introducción
En este laboratorio  se observo cómo se comportan las señales tanto en el tiempo como en la frecuencia. Lo haremos aplicando tres técnicas fundamentales: la convolución, la correlación y la transformada de Fourier. Además del análisis de una señal electrooculograma (EOG).

## importación de librerias 
```python
!pip install wfdb
import matplotlib.pyplot as plt
import numpy as np
import wfdb
import pandas as pd
import os
from scipy.stats import norm
import seaborn as sns
from scipy.fft import fft, fftfreq
from scipy.signal import welch
```

Este bloque importa librerías clave para analizar señales biológicas: ` wfdb`  para leer datos fisiológicos, ` numpy`  y `  pandas`  para manejo numérico y de datos, ` matplotlib`  y `  seaborn`  para gráficos, y funciones de `scipy` para calcular la transformada de Fourier y la densidad espectral de potencia, herramientas fundamentales para estudiar las características temporales y frecuenciales de la señal.

<h1 align="center"><i><b>PARTE A DEL LABORATORIO</b></i></h1>

<img width="678" height="442" alt="image" src="https://github.com/user-attachments/assets/43539f60-24df-4e2b-87d7-fad9f343d62a" />

<img width="699" height="440" alt="image" src="https://github.com/user-attachments/assets/3acd7d9a-fb69-427e-ba16-9345ba6500d0" />

## Señal y[n] resultante de la convolución en python

```python

import numpy as np

# Señales
h = np.array([5,6,0,0,8,2,8,5,6,0,0,8,1,6])
x = np.array([1,0,7,2,6,4,3,8,0,7,1,0,1,9,9,8,8,7,1,4])

# Convolución
y = np.convolve(x, h)

print("y[n] =", y)
```

 **resultados**

y[n] = [  5   6  35  52  50  58 103  93 162 130 179 158 102 272 202 292 189 268
 242 211 331 250 223 192 172 166 153 150 112  63  75  10  24]

 ```python
# Tus datos
h = np.array([5,6,0,0,8,2,8,5,6,0,0,8,1,6])
x = np.array([1,0,7,2,6,4,3,8,0,7,1,0,1,9,9,8,8,7,1,4])

y = np.array([5, 6, 35, 52, 50, 58, 103, 93, 162, 130, 179, 158,
              102, 272, 202, 292, 189, 268, 242, 211, 331, 250,
              223, 192, 172, 166, 153, 150, 112, 63, 75, 10, 24])

# Ejes
nx = np.arange(len(x))
nh = np.arange(len(h))
ny = np.arange(len(y))

# ================================
# Figura horizontal
# ================================
plt.figure(figsize=(18,5))

# -------- x[n] --------
plt.subplot(1,3,1)
plt.stem(nx, x, linefmt='purple', markerfmt='o', basefmt=" ")
plt.title('Señal x[n]')
plt.xlabel('n')
plt.ylabel('x[n]')
plt.grid()

# -------- h[n] --------
plt.subplot(1,3,2)
plt.stem(nh, h, linefmt='purple', markerfmt='o', basefmt=" ")
plt.title('Señal h[n]')
plt.xlabel('n')
plt.ylabel('h[n]')
plt.grid()

# -------- y[n] --------
plt.subplot(1,3,3)
plt.stem(ny, y, linefmt='purple', markerfmt='o', basefmt=" ")
plt.title('Convolución y[n]')
plt.xlabel('n')
plt.ylabel('y[n]')
plt.grid()

plt.tight_layout()
plt.show()

```

<img width="1790" height="490" alt="image" src="https://github.com/user-attachments/assets/39764ec7-6014-40c5-b80c-90d7871ac808" />


Este código en Python calcula la convolución discreta entre dos señales utilizando la función `np.convolve()` de` NumPy`. Primero, se definen dos listas, `h` y `x`, que representan la respuesta al impulso de un sistema y una señal de entrada, respectivamente. Luego, se aplica la convolución entre estas dos señales usando `np.convolve(x, h, mode='full')`, lo que genera una nueva señal y cuya longitud es la suma de las longitudes de `x` y `h` menos uno. 

La convolución es una operación fundamental en procesamiento de señales, ya que permite analizar cómo una señal se ve afectada por un sistema. Finalmente, el código imprime las señales `h`, `x` y y para visualizar los datos y el resultado de la convolución.

<h1 align="center"><i><b>PARTE B DEL LABORATORIO</b></i></h1>

```python
import numpy as np
import matplotlib.pyplot as plt

# Parámetros
Ts = 1.25e-3
f = 100
n = np.arange(0,9)

# Señales
x1 = np.cos(2*np.pi*f*n*Ts)
x2 = np.sin(2*np.pi*f*n*Ts)

# Correlación cruzada
r12 = np.correlate(x1, x2, mode='full')
lags = np.arange(-len(x1)+1, len(x1))

plt.figure(figsize=(15,4))

# x1
plt.subplot(1,3,1)
plt.stem(n, x1, linefmt='purple', markerfmt='o', basefmt='r-')
plt.title("x1[n] = cos(2π100nTs)")
plt.xlabel("n")
plt.ylabel("Amplitud")
plt.grid()

# x2
plt.subplot(1,3,2)
plt.stem(n, x2, linefmt='purple', markerfmt='o', basefmt='r-')
plt.title("x2[n] = sin(2π100nTs)")
plt.xlabel("n")
plt.ylabel("Amplitud")
plt.grid()

# correlación
plt.subplot(1,3,3)
plt.stem(lags, r12, linefmt='purple', markerfmt='o', basefmt='r-')
plt.title("Correlación cruzada entre x1[n] y x2[n]")
plt.xlabel("Retardo k")
plt.ylabel("r12[k]")
plt.grid()

plt.tight_layout()
plt.show()
```

<img width="1490" height="390" alt="image" src="https://github.com/user-attachments/assets/f21266fd-7d19-47fe-9105-73eb39bdc5a2" />



<h1 align="center"><i><b>PARTE C DEL LABORATORIO</b></i></h1>
inicalmente para la adquisición de la señal EOG se utilizó el código proporcionado que emplea la librería `nidaqmx` , la cual permite interactuar con dispositivos NI DAQ para la captura de señales analógicas. En el código se configura el canal de entrada analógica, la frecuencia de muestreo (800 Hz, cumpliendo el criterio de Nyquist), y el tiempo total de adquisición (5 segundos). Luego, se realiza la lectura finita de muestras y se guarda la señal en un vector. Finalmente, se genera un gráfico que muestra la señal adquirida en función del tiempo, permitiendo visualizar claramente la señal EOG en formato digital lista para su posterior análisis.

```python
Librería de uso de la DAQ
!python -m pip install nidaqmx     

Driver NI DAQ mx
!python -m nidaqmx installdriver   

Created on Thu Aug 21 08:36:05 2025
@author: Carolina Corredor
"""

# Librerías: 
import nidaqmx                     # Librería daq. Requiere haber instalado el driver nidaqmx
from nidaqmx.constants import AcquisitionType # Para definir que adquiera datos de manera consecutiva
import matplotlib.pyplot as plt    # Librería para graficar
import numpy as np                 # Librería de funciones matemáticas

#%% Adquisición de la señal por tiempo definido

fs = 800           # Frecuencia de muestreo en Hz. Recordar cumplir el criterio de Nyquist
duracion = 5       # Periodo por el cual desea medir en segundos
senal = []          # Vector vacío en el que se guardará la señal
dispositivo = 'Dev3/ai0' # Nombre del dispositivo/canal (se puede cambiar el nombre en NI max)

total_muestras = int(fs * duracion)

with nidaqmx.Task() as task:
    # Configuración del canal
    task.ai_channels.add_ai_voltage_chan(dispositivo)
    # Configuración del reloj de muestreo
    task.timing.cfg_samp_clk_timing(
        fs,
        sample_mode=AcquisitionType.FINITE,   # Adquisición finita
        samps_per_chan=total_muestras        # Total de muestras que quiero
    )

    # Lectura de todas las muestras de una vez
    senal = task.read(number_of_samples_per_channel=total_muestras)

t = np.arange(len(senal))/fs # Crea el vector de tiempo 
plt.plot(t,senal)
plt.axis([0,duracion,-0.7,0.11])
plt.grid()
plt.title(f"fs={fs}Hz, duración={duracion}s, muestras={len(senal)}")
plt.show()
```

```python
import numpy as np
import matplotlib.pyplot as plt

N = len(senal)
fs = 1000
t = np.arange(N) / fs

plt.figure(figsize=(12,6))
plt.plot(t, senal, color='purple')
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("Señal del generador EOG")
plt.grid()
plt.show()
```

<img width="1012" height="547" alt="image" src="https://github.com/user-attachments/assets/e82aaf11-fa4a-4c23-885b-37453f63f742" />

Este código carga una señal EOG desde un archivo `CSV `usando `pandas`, extrae las columnas de `tiempo` y `voltaje`, y luego grafica la señal en función del tiempo con `matplotlib`, mostrando la variación del voltaje en milivoltios y agregando etiquetas y una cuadrícula para facilitar la visualización.
*grafica*


```python
fs = 1000
N = len(senal)

Y = np.fft.fft(senal)
frecuencias = np.fft.fftfreq(N, d=1/fs)

freq_pos = frecuencias[:N//2]
magnitud = np.abs(Y[:N//2])

f_dominante = freq_pos[np.argmax(magnitud)]
f_nyquist = 2 * f_dominante

print("Frecuencia dominante:", f_dominante)
print("Frecuencia de Nyquist:", f_nyquist)
```
se calcularon la frecuencia de muestreo de 4 veces la frecuencia de Nyquist

 **resultados**
 
- Frecuencia dominante: 2.0
- Frecuencia de Nyquist: 4.0
- Nueva frecuencia de muestreo: 16.0 Hz

<img width="1012" height="547" alt="image" src="https://github.com/user-attachments/assets/4195b9b9-b5d9-407b-b022-efd5160b537e" />

Se calcularon los estadísticos descriptivos fundamentales de la señal EOG para caracterizar su comportamiento en el dominio temporal. 
*La` media `indica el valor promedio general.

*la `mediana` representa el punto central de los datos.

*la `desviación` estándar muestra la variabilidad o dispersión de la señal.

 *El `máximo` y `mínimo` reflejan los valores extremos o picos. 

 ```python
media = np.mean(senal)
mediana = np.median(senal)
desv = np.std(senal)
maximo = np.max(senal)
minimo = np.min(senal)

print("Media:", media)
print("Mediana:", mediana)
print("Desviación estándar:", desv)
print("Máximo:", maximo)
print("Mínimo:", minimo)
```
 Estos parámetros son esenciales para comprender la distribución y estabilidad de la señal antes de realizar análisis más profundos.

 
 **resultados**
 
- Media: -0.03602852356785734
- Mediana: -0.03468168468680233
- Desviación estándar: 0.1436685720789419
- Máximo: 0.40822401049081236
- Mínimo: -0.5458257573191077

Se calcula la Transformada de Fourier para analizar la señal en frecuencia, mostrando solo las frecuencias positivas hasta 100 Hz para mayor claridad. Además, se estima la densidad espectral de potencia con el método de Welch y se grafican ambos resultados para visualizar la distribución de energía en la señal.

 ```python
fs = 1000
N = len(senal)

Y = np.fft.fft(senal)
frecuencias = np.fft.fftfreq(N, d=1/fs)

freq_pos = frecuencias[:N//2]
magnitud = np.abs(Y[:N//2])

plt.figure(figsize=(12,6))
plt.plot(freq_pos, magnitud, color='purple')
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Transformada de Fourier de la Señal")
plt.grid()
plt.show()
```

<img width="1005" height="547" alt="image" src="https://github.com/user-attachments/assets/c73bdfb3-b7a2-4968-b97a-4211cf963bb8" />

 ahora su densidad espectral de potencia
 

 ```python
PSD = (np.abs(Y)**2) / N
PSD_pos = PSD[:N//2]

plt.figure(figsize=(12,6))
plt.plot(freq_pos, PSD_pos, color='purple', linewidth=2)
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Potencia")
plt.title("Densidad Espectral de Potencia (PSD)")
plt.grid()
plt.show()
```
<img width="996" height="547" alt="image" src="https://github.com/user-attachments/assets/549277b8-5676-4dfa-bee0-d6474d0738b0" />

como observamos que las graficas en tiempo de frecuencia era muy grande, lo que hicimos fue escoger una ventana de tiempo de 0 a 20 hz, ya que esa es la frecuencia de una señal EOG

 ```python
plt.figure(figsize=(12,6))

plt.plot(freq_pos, magnitud, color='purple', linewidth=2)

plt.xlim(0, 20)
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Transformada de Fourier (0–20 Hz)")
plt.grid()
plt.show()
```

<img width="1020" height="547" alt="image" src="https://github.com/user-attachments/assets/ca07c284-ecef-40a9-9239-1dd937635263" />

```python
plt.figure(figsize=(12,6))

plt.plot(freq_pos, PSD_pos, color='purple', linewidth=2)

plt.xlim(0, 20)
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Potencia")
plt.title("Densidad Espectral de Potencia (0–20 Hz)")
plt.grid()
plt.show()
```

<img width="1012" height="547" alt="image" src="https://github.com/user-attachments/assets/dd03bb5c-7092-41aa-b9d4-45a038de67f4" />

A partir de la PSD se obtienen tres estadísticas clave:

*la `frecuencia media` es el promedio ponderado por la potencia.*

*la `frecuencia mediana` divide la energía total en dos partes iguales.*

*la `desviación estándar`mide la dispersión de la energía en frecuencias.*

posteriormente, se grafica un histograma que muestra la distribución de potencia a lo largo del espectro de frecuencias, facilitando la visualización de dónde se concentra la energía de la señal.
```python
N = len(senal)

# FFT
X = np.fft.fft(senal)
frecuencias = np.fft.fftfreq(N, 1/fs)

# Solo parte positiva
pos_mask = frecuencias >= 0
frecuencias_pos = frecuencias[pos_mask]
PSD = (1/N) * np.abs(X[pos_mask])**2

# Normalizamos la PSD para usarla como peso
PSD_norm = PSD / np.sum(PSD)

# -----------------------------
# i. Frecuencia media
frecuencia_media = np.sum(frecuencias_pos * PSD_norm)

# ii. Frecuencia mediana
cumulative = np.cumsum(PSD_norm)
frecuencia_mediana = frecuencias_pos[np.where(cumulative >= 0.5)[0][0]]

# iii. Desviación estándar
desviacion_frecuencia = np.sqrt(
    np.sum(((frecuencias_pos - frecuencia_media)**2) * PSD_norm)
)

# Mostrar resultados
print("Frecuencia media:", frecuencia_media, "Hz")
print("Frecuencia mediana:", frecuencia_mediana, "Hz")
print("Desviación estándar:", desviacion_frecuencia, "Hz")
```
**resultados**
- Frecuencia media: 9.497832446885091 Hz
- Frecuencia mediana: 4.0 Hz
- Desviación estándar: 32.11530631233756 Hz

<img width="1389" height="490" alt="image" src="https://github.com/user-attachments/assets/2dfce110-b8dc-4ec5-b6f6-3f2e3e360b71" />

Adicionalmente se realizo la clasifiacion de la señal :

**Determinística o Aleatoria:** La señal EOG es generalmente aleatoria, ya que presenta variaciones impredecibles debido a la actividad natural del ojo y el ruido biológico, aunque puede tener componentes periódicos asociados a movimientos repetitivos.

**Periódica o Apperiodica:** La señal EOG es apperiodica o no estrictamente periódica, dado que sus características no se repiten de forma exacta en el tiempo, reflejando la dinámica irregular de los movimientos oculares.

**Analógica o Digital:** Originalmente, la señal EOG es analógica, ya que es una señal continua en el tiempo y en amplitud. Sin embargo, al ser adquirida y almacenada en un computador mediante un proceso de muestreo, se convierte en una señal digital para su procesamiento.

<h1 align="center"><i><b>Bibliografia</b></i></h1>

[1]https://docs.github.com/es/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax

[2]https://www.w3schools.com/python/default.asp

[3]https://www.spyder-ide.org/

[4]https://www.ni.com/es/support/downloads/drivers/download.ni-daq-mx.html?srsltid=AfmBOop-UElW8OLmEl4SfkDFLwaI0hCgohxlfwcTWo326OHB-MLJ-aNv#569353

[5]https://www.anaconda.com/download

