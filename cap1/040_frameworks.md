---
title: Grafos de computación y frameworks
subject: Aprendizaje Profundo
subtitle: Llevando la teoría a código
short_title: Grafos de computación
authors:
  - name: Jorge Anais
    orcid: 0000-0001-9051-1338
    email: jrganais@gmail.com
license: MIT
keywords: keras, pytorch, python
---

## Redes directas totalmente conectadas 

Considera el diagrama de la [](#mlp_2_5_3_1_clean). 
A este tipo de redes también se les llama **redes directas totalmente conectadas** o en inglés: ***fully connected feed forward networks***. 
Esto hace referencia a que todas las neuronas de una capa están conectadas con todas las neuronas de la capa anterior, y a todas las neuronas de la capa siguiente. 
Eso sí, fíjate que las neuronas de una misma capa no están conectadas entre sí. 


:::{figure} ../_images/mlp_2_5_3_1_clean.svg
:label: mlp_2_5_3_1_clean
Red neuronal con dos capas ocultas con 5 y 3 neuronas respectivamente.
:::


:::{important}
**Importante**

El número de parámetros que tiene una red neuronal, es decir, la cantidad de pesos y sesgos depende del número de neuronas de cada capa. Una propiedad importante de las redes neuronales es su capacidad. En general, entre más parámetros tenga una red neuronal, mayor será su capacidad. 
:::


:::{tip}
**Ejercicio**

Calculemos cuantos parámetros tiene la red de la [](#mlp_2_5_3_1_clean), es decir, cuántos pesos y sesgos tiene en total. La capa de entrada tiene dos valores, que entran a la primera capa oculta. La primera capa oculta tiene 5 neuronas. Cada neurona tiene dos pesos, uno para cada valor de entrada y un sesgo, es decir, cada neurona tiene 3 parámetros. Multiplicando obtenemos que la primera capa tiene 15 parámetros en total. 

Ahora calculemos los parámetros de la segunda capa oculta. Cada neurona de esta capa recibe 5 valores (uno por cada neurona de la capa anterior), por lo tanto, debe tener 5 parámetros de peso más un parámetro de sesgo. Multiuplicando por las 3 neuronas obtenemos que la segunda capa oculta tiene 18 parḿetros en total. 

Finalmente tenemos la capa de salida, la cual tiene una única neurona. Esta recibe 3 valores de la capa previa, por lo tanto tiene 3 pesos más un sesgo, totalizando 4 parámetros.

Sumando todo lo anterior, tenemos que la red de la figura tiene en total $15+18+4=37$ parármetros.
:::


## Bibliotecas de aprendizaje profundo

Si bien es perfectamente posible escribir un código en Python "puro" que permita emular el comportamiento de un perceptrón multicapa (ver por ejemplo https://youtu.be/W8AeOXa_FqU?si=a_ud1WaxVB6MX2uo), notaremos que a medida que las arquitecturas de las redes se vuelven más complejas y necesitemos muchos datos para entrenar a las redes, las cosas se comienzan a complicar. Existen diferentes frameworks de programación que permiten abstraer esta complejidad. Consideremos los siguientes niveles de abstracción:


:::{important}
**Importante**

- Nivel 0: Programación en Python puro (poca abstracción)
- Nivel 1: Framework de diferenciación automática: [Tensorflow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/)
- Nivel 2: Composición de capa: [Keras](https://keras.io/), PyTorch
- Nivel 3: Modelos completamente definidos: [ScikitLearn](https://scikit-learn.org/stable/) (nivel más alto de abstracción)
:::

Nota que a medida que subimos en nivel de abstracción, la cantidad de código necesario para hacer redes complejas disminuye, pero así también disminuye la flexibilidad para crear arquitecturas de redes a medida. 

En este curso navegaremos entre los niveles de abstracción 1 y 2. Siéntete libre de explorar la documentación asociada a estas herramientas.


## Introducción a PyTorch y grafos de computación

PyTorch es un framework de diferenciación automática, eso quiere decir, que dado un grafo de computación, PyTorch puede calcular automáticamente las derivadas y aplicar la retropropagación del gradiente. 

Un grafo esta computesto por nodos unidos por aristas, y nos permite representar un calculo en sus partes más elementales.

Para ilustrar este concepto, considere la siguiente composición de funciones:

\begin{align*}
u &= x * a \\
v &= \log(u) \\
w &= v + b \\
t &= 1/w \\
z &= t - k \\
y &= z^2
\end{align*}

La [](#grafo_v2) muestra el  grafo de computación equivalente a la composición de funciones anteriores. Notamos que las variables de entrada son $x$ y $a$, las cuales primero se multiplican, al resultado se le aplica la función logaritmo, luego se le suma el valor $b$ . Se calcula el inverso del resultado anterior y le restamos el valor $k$. Finalmente se eleva al cuadrado y obtenemos el valor final $y$.


:::{figure} ../_images/grafo_v2.png
:label: grafo_v2
Grafo de computación
:::

:::{tip}
**Pregunta**

¿Cómo luciría el grafo de computación de un perceptrón multicapa? Intenta realizar uno. 
:::


## Link de interés

Utiliza el simulador de redes neuronales disponible en  https://playground.tensorflow.org/. Prueba con diferentes conjuntos de datos, diferentes capas de entrada, capas ocultas, neuronas por capa. Observa que pasa al aumentar sistematicamente el número de neuronas, y el número de capas.


## Revisemos lo aprendido

Analiza el caso presentado y elige la alternativa que mejor refleje lo aprendido esta semana: 


::::{tip}
**Pregunta**

Valentina es estudiante de ingeniería y está construyendo su primera red neuronal para clasificar flores del dataset Iris. Su red tiene una capa de entrada con 4 neuronas (largo y ancho de sépalo y pétalo), una única capa oculta con 6 neuronas usando activación ReLU, y una capa de salida con 3 neuronas (una por especie). Antes de entrenarla, decide contar el total de parámetros de su modelo para estimar su capacidad. 
¿Cuántos parámetros (pesos y sesgos) tiene en total la red neuronal de Valentina? 

A) La red tiene 33 parámetros: 24 pesos y 9 sesgos.  
B) La red tiene 45 parámetros: 42 pesos y 3 sesgos.  
C) La red tiene 51 parámetros: 48 pesos y 3 sesgos.  
D) La red tiene 39 parámetros: 36 pesos y 3 sesgos.  
E) No es posible calcularlo sin conocer la función de activación de la capa de salida.  

:::{hint}
**Respuesta** 

La pregunta evalúa el cálculo de parámetros de una red totalmente conectada, que es uno de los conceptos clave de la semana. La alternativa correcta es C (51 parámetros), calculados así: Capa oculta: 6 neuronas × (4 pesos + 1 sesgo) = 30 parámetros. Capa de salida: 3 neuronas × (6 pesos + 1 sesgo) = 21 parámetros. Total: 51. Los distractores cubren errores frecuentes: olvidar los sesgos de la capa oculta (D), mezclar pesos y sesgos (A y B), o pensar que la función de activación afecta el conteo de parámetros (E). 
:::
::::

## **Lectura complementaria** 

Para profundizar en el contenido revisa el libro disponible gratuitamente *Alice’s Adventures in a differentiable wonderland* por Simone Scardapane (en adelante Scardapane 2024,  https://doi.org/10.48550/arXiv.2404.17625). En particular el capitulo 2 permite ahondar en los conceptos matemáticos vistos esta semana. 

Para revisar conceptos con una mirada práctica se recomienda el libro *Hands-on machine learning with Scikit-Learn and PyTorch* por Aurélien Géron (en adelante Géron 2025, https://learning-oreilly-com.webezproxy.duoc.cl/library/view/hands-on-machine-learning/9798341607972/ ). El capítulo 9 introduce los conceptos asociados a redes neuronales artificiales.

