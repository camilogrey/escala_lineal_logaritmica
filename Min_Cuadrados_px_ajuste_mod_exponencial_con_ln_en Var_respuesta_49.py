"""minimos cuadrados para ajustar el modelo de crecimiento exponencial tomando 
el logaritmo de la variable de respuesta"""
import matplotlib.pyplot as plt
import numpy as np

# Datos
t = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
P = np.array([100, 250, 625, 1563, 3906, 8056, 17000, 41506, 94400, 200001])

# Gráfico en escala lineal
plt.plot(t, P, 'o')
plt.xlabel('Tiempo (horas)')
plt.ylabel('Población')
plt.title('Gráfico en escala lineal')
plt.show()

"""El gráfico en escala lineal muestra una curva de crecimiento exponencial 
pero es difícil de ajustar a un modelo exponencial debido a la curvatura de la línea"""
# Gráfico en escala logarítmica
plt.plot(t, np.log(P), 'o')
plt.xlabel('Tiempo (horas)')
plt.ylabel('Logaritmo de la población')
plt.title('Gráfico en escala logarítmica')
plt.show()

"""En contraste, el gráfico en escala logarítmica muestra una relación lineal clara entre el tiempo 
y el logaritmo de la población, Podemos ajustar un modelo lineal a estos datos y luego transformar 
el modelo en un modelo exponencial."""

import statsmodels.api as sm

# Modelo logarítmico
log_P = np.log(P)
X = sm.add_constant(t)
model = sm.OLS(log_P, X).fit()

# Parámetros del modelo
a = np.exp(model.params[0])
b = model.params[1]

print(f'Parámetros del modelo: a={a:.2f}, b={b:.2f}')

# Predicción de la población en tiempo 6
t_new = 6
P_new = a * np.exp(b * t_new)
print(f'Predicción de la población en tiempo {t_new}: {P_new:.2f}')
