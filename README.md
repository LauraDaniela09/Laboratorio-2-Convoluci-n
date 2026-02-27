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

import matplotlib.pyplot as plt

n = np.arange(len(y))

plt.figure()
plt.stem(n, y)
plt.xlabel('n')
plt.ylabel('y[n]')
plt.stem(n, y, linefmt='purple', markerfmt='o', basefmt=" ")
plt.title('Señal de la Convolución y[n] = x[n] * h[n]')
plt.grid()
plt.show()

```

<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/bef099b4-29d9-4616-9d06-74f17d9d1d80" />

Este código en Python calcula la convolución discreta entre dos señales utilizando la función `np.convolve()` de` NumPy`. Primero, se definen dos listas, `h` y `x`, que representan la respuesta al impulso de un sistema y una señal de entrada, respectivamente. Luego, se aplica la convolución entre estas dos señales usando `np.convolve(x, h, mode='full')`, lo que genera una nueva señal y cuya longitud es la suma de las longitudes de `x` y `h` menos uno. 

La convolución es una operación fundamental en procesamiento de señales, ya que permite analizar cómo una señal se ve afectada por un sistema. Finalmente, el código imprime las señales `h`, `x` y y para visualizar los datos y el resultado de la convolución.

<h1 align="center"><i><b>PARTE B DEL LABORATORIO</b></i></h1>
