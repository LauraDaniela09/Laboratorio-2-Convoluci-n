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
plt.title("Señal del generador - Dominio del tiempo")
plt.grid()
plt.show()
```

<img width="1012" height="547" alt="image" src="https://github.com/user-attachments/assets/ce496940-db39-4522-b1cc-045de3e29ffe" />
