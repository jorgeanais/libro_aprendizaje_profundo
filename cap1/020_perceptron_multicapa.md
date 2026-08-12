---
title: El perceptrón multicapa
subject: Aprendizaje Profundo
subtitle: Haciendo más profundo el modelo
short_title: Perceptrón multicapa
authors:
  - name: Jorge Anais
    orcid: 0000-0001-9051-1338
    email: jrganais@gmail.com
license: MIT
keywords: perceptron, funcion de activación
---

Al igual que el cerebro tiene muchos muchísimas conexiónes, del órden de $10^{14}$ o cientos de billones -en castellano como diría mi abuelita- unir muchos perceptrones en capas permite establecer lo que conocemos como perceptrón multicapa. La [](#deep_neural_network) muestra un esquema de un perceptrón multicapa


:::{figure} ../_images/deep_neural_network.svg
:label: deep_neural_network
Representación de un perceptrón multicapa.
:::

Las partes de un perceptrón multicapa son las siguientes:

- **Capa de entrada**: Corresponde a las variables de entrada de la red
- **Capas ocultas**: Son todas las capas que están definidas entre la capa de entrada y la de salida. El número de capas y la cantidad de "neuronas" son elegidas a mano. Notar que cada capa oculta tiene sus respectivos pesos, sesgo y función de activación.
- **Capas de salida**: Corresponde a la capa final, que entrega el resultado de la red. Puede ser un único valor, o en el caso de clasificación, puede corresponder a varios números (uno por cada clase a predecir). Notar que la capa de salida también tiene parámetros $U$ para los pesos, $c$ para el sesgo y $g$ para la función de activación.



Para hacer la notación más simple la notación matemática, una red se puede representar en notación matricial, de modo que cada capa se representará como una matriz. 


:::{figure} ../_images/perceptron_ejemplo_1.png
:label: perceptron_ejemplo
Perceptrón con una capa de entrada, una oculta y capa de salida.
:::

## Compactando la escritura

Calculemos las salidas de la primera capa oculta del perceptrón de la [](#perceptron_ejemplo), es decir, los valores de la neurona de arriba ($h_1$) y la neurona de abajo ($h_2$) de la primera capa.

\begin{align}
h_1 &= \text{ReLu}(x_1 \cdot w_{11} + x_2 \cdot w_{12} + b) \\
h_2 &= \text{ReLu}(x_1 \cdot w_{21} + x_2 \cdot w_{22} + b)
\end{align}

Utilizando algebra lineal, podemos escribir esto mismo como una multiplicación matricial

$$
(h_1 \quad h_2) = \text{ReLu} \left( (x_1 \quad x_2) \begin{pmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \end{pmatrix} + (b_1 \quad b_2) \right)
$$

## Representación general de una red neuronal

En general, una red neuronal se puede escribir matemáticamente como una composición de funciones. Esto es importante, ya que veremos que los *frameworks* de aprendizaje automático se basan en esta cualidad, que llamaremos grafos de computación y que se discuten más adelante. Por ahora, concentremonos en la [](#composicion):

:::{figure} ../_images/matricial.png
:label: composicion
Representación de un perceptrón multicapa como una composición de funciones escritas de manera vectorial.
:::


Donde $x$ es el vector con los valores de la capa de entrada, $W$ es la matriz con todos los pesos de la capa oculta, $b$ es el vector con los sesgos de la capa oculta y $f$ es la función de activación que se aplica a la capa oculta. Luego, para calcular el valor de la salida, se multiplican los valores obtenidos en la capa oculta (vector $h$) por la matriz de pesos $U$ de la capa de salida, se le suma el sesgo $c$ y se aplica la función de activación $g$ de la capa de salida.

Si tenemos más que dos capas ocultas, simplemente se componen de tal manera que la salida de la anterior capa es la entreda de la siguiente capa, como se ilustra en [](#mlp_perspective_layers).

:::{figure} ../_images/mlp_perspective_layers.svg
:label: mlp_perspective_layers
Representación de una red con $L$ capas ocultas. La información fluye de una capa a la siguiente. Los pesos de cada capa están representados por las matrices $W^i$. Los pesos de la capa de salida están representados por la matriz $U$. Fuente: Elaboración propia.
:::

:::{important}
Notar que hay *hardware* especializado como GPUs que permite realizar operaciones matriciales extremadamente rápido, gracias a potentes librerías de bajo nivel como CUDA que han sido optimizadas al máximo para sacar el mayor provecho al momento de hacer este tipo de operaciones matemáticas, especialmente gracias a la *paralelización*. Ahora revisa el cuaderno interactivo a continuación "**Comparando GPU vs CPU**".
:::