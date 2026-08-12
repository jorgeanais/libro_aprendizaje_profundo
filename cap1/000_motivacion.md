---
title: Introducción
subject: Aprendizaje Profundo
subtitle: Construyendo redes neuronales simples para tareas complejas
short_title: Intro
authors:
  - name: Jorge Anais
    orcid: 0000-0001-9051-1338
    email: jrganais@gmail.com
license: MIT
keywords: perceptron, funcion de activación
---

El mundo que nos rodea tiene patrones, los cuales los humanos hemos ido aprendiendo a dominar con el tiempo. Por ejemplo, hemos comprendido los ciclos de las mareas, sus causas y esto nos ha permitido hacer predicciones sobre cual es el mejor momento para la pesca o surfear las olas. 

Es importante comprender las variables que influyen sobre los fenómenos: hora del día, lugar, posición de la Luna  y el Sol relativa a la Tierra, y como se relacionan con la variable a predecir: nivel del mar.  Todo esto se puede modelar mediante ecuaciones matemáticas, que dado las variables de entrada, entregan un valor estimado de la variable a predecir.

:::{figure} ../_images/1280px-Great_Wave_off_Kanagawa2.jpg
:label: Great_Wave_off_Kanagawa
Katsushika Hokusai, Great Wave off Kanagawa. Dominio público, https://commons.wikimedia.org/w/index.php?curid=5576388
:::

Otro ejemplo más cercano, puede ser como los seres humanos identificamos los objetos que nos rodean. En palabras sencillas, nuestros ojos reciben luz del ambiente, la cual es enfocada por el cristalino en nuestra retina. En las células de la retina se produce una señal electro-química que viaja por el sistema nervioso hasta nuestro cerebro, quien finalmente interpreta esas señales identificando los objetos que están a nuestro alrededor. 

Imaginemos por un momento que podemos modelar este fenómeno complejo del reconocimiento de objetos como una función matemática. La entrada de esta función es la luz que reciben nuestros ojos, y la variable de salida corresponde al reconocimiento del tipo de objeto, por ejemplo: un perro. ¿Cómo podría ser esta función matemática? quizás es un polinomio muy complejo, o quizás sea una intrincada fórmula trigonométrica, o peor aún, una combinación de todo lo anterior. Es muy dificil saberlo, considerando la complejidad de todos los procesos que ocurren: el sistema óptico para concentrar la luz en la retina, la interacción de la luz con las células del ojo, la química de los impulsos que viajan a través del sistema nervioso, el procesamiento de esa información en el cerebro, entre otros procesos que están ocurriendo simultáneamente.


:::{figure} ../_images/visionhumana.png
:label: visionhumana
Obtenido de https://www.clinicabaviera.com/blog/quieres-saber-como-se-produce-la-vision/
:::

Es curioso que realmente no necesitemos saber  la ecuación exacta de este fenómeno para entrenar una máquina que haga lo mismo. De hecho, solo nos basta con una función que sea suficientemente parecida, tan parecida que nos entregue prácticamente el mismo resultado esperado.

Las redes neuronales artificiales son un modelo de aprendizaje de máquinas que nos permite acercarnos tanto como podamos a una función objetivo. Eso sí,  siempre y cuando la red sea suficientemente grande y tengamos suficientes ejemplos para entrenar a la red.

El Teorema de Aproximación Universal establece que una red neuronal artificial suficientemente grande (infinita), puede aproximar cualquier función contínua con precisión arbitraria. Es decir, podemos modelar aproximadamente fenómenos naturales sin importar su complejidad, siempre que tengamos suficientes datos y la suficiente capacidad de computo para almacenar y entrenar la red.



## El Perceptrón 

La inspiración para el desarrollo de las redes neuronales aritificiales, viene, como su nombre lo indica, del estudio de las neuronas. Cada neurona recibe señales eléctroquímicas de sus neuronas vecinas a través de fibras llamadas dendritas. Cuando la suma total de las señales electroquímicas entrantes supera un cierto umbral, la neurona se activa, transmitiendo una señal a sus vecinas.

El primer modelo matemático de una neurona fue introducido por Frank Rosenblatt en 1957  y se llamó perceptrón. La figura # presenta un esquema de un perceptrón.

:::{figure} ../_images/perceptron_model.svg
:label: perceptron_model
Representación esquemática de un perceptrón.
:::


En el esquema de la [](#perceptron_model), los valores $x_i$ representan los parámetros de entrada (valores numéricos), los pesos $w_i$ multiplican los valores de entrada y posteriormente todos estos valores se suman junto con el valor $b$ llamado sesgo (bias en inglés). El valor numérico resultante se pasa a través de la llama función de activación $f$, que representa si la neurona se ha activado o no. En esta analogía, un valor pequeño de salida representa que no se ha activado, mientras que un valor grande representa que si se ha activado.

## Función de activación

Algunas funciones de activación típicas son la función sigmoídea $\sigma(u)$, la función tangente hiperbólica $\tanh(u)$, la función de rectificación lineal $\text{ReLu}(u)$, entre otras. Notar que la elección de la función de activación tiene implicancias durante del entrenamiento de la red, lo que veremos más adelante en el curso.


:::{figure} ../_images/funciones_activacion.png
:label: funciones_activacion
Funciones de activación típicas.
:::

:::{important}
La función de activación introduce la **no linealidad** necesaria para que la red pueda aprender patrones complejos, ya que sin ella, la combinación matemática de múltiples capas colapsaría en una simple operación lineal equivalente a una red de una sola capa.
:::

## Ejemplo de un perceptrón


La [](#perceptron_model) representa un perceptrón con una capa de entrada, una capa oculta con dos neuronas, y una capa de salida con una única neurona. Matemáticamente esto lo escribimos paso a paso de la siguiente manera:

$$ u = w_1 x_1 + w_2 x_2 + w_3 x_3 + b$$

$$ y = f(u)$$

Recuerda que $x_i$ representan los valores de la entrada, $w_i$ y $b$ son los parámetros del perceptrón y $f$ representa la función de activación. Finalmente, $y$ es el valor resultante o salida.


::::{tip}
**Ejercicio**

Considere que los parámetros valen $w_1 = 1$, $w_2 = -1$ , $w_3 = 2$ y $b=3$, y que se utiliza la función de activación  $f(u)=\text{ReLU}(u)$. ¿Cuál es el valor de la salida $y$ del perceptrón si los valores de la entrada son $x_1 = x_2 =x_3 = 1$?
:::{hint}
**Respuesta**

Primero calculamos $$ u = 1 \cdot 1 + (-1) \cdot 1 + 2 \cdot 1 + 3 = 5$$ luego aplicamos la función de activación $$ y = f(5) = 5$$ Por lo tanto, el valor de salida $y$ del perceptrón es 5.
:::
::::


## Pon en práctica el conocimiento

En este curso utilizaremos cuadernos interactivos **Jupyter notebook** que permitiran ejecutar de manera interactiva código escrito en Python.

Como primer ejercicio, realiza la actividad contenida en el cuaderno interactivo "Un perceptrón en Python".
