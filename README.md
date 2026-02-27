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
