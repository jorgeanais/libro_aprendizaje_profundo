---
title: Arquitectura de una red neuronal
subject: Aprendizaje Profundo
subtitle: Como diseñar y entrenar una red
short_title: Arquitectura
authors:
  - name: Jorge Anais
    orcid: 0000-0001-9051-1338
    email: jrganais@gmail.com
license: MIT
keywords: datos, entrenamiento, arquitectura
---


Hasta ahora hemos discutido los componentes básicos de una red neuronal directa totalmente conectada, que incluía capas de entrada, oculta y de salida, que aportan a número de parámetros (pesos y sesgos de la red).  Simbolizaremos a todo este conjunto de parámetros mediante la letra griega $\theta$.

Además, tenemos flexibilidad en definir la arquitectura de la red, es decir, la red puede tener tantas capas y neuronas como queramos (o quepan en la memoria de nuestra computadora). También podemos elegir la función de activación para nuestras capas ocultas. Estas características arbitrarias de la red se llaman hiperparámetros, y se diferencian de los parámetros de la red (pesos y sesgos) porque los hiperparámetros quedan fijos al momento de definir la red, y no se modifican posteriormente.

En esta sección nos centraremos en el proceso de hemos discutido los componentes básicos de una red neuronal directa totalmente conectada, que incluía capas de entrada, oculta y de salida, que aportan a número de parámetros (pesos y sesgos de la red).  Simbolizaremos a todos los parámetros de una red mediante la letra griega $\theta$.

Además, tenemos flexibilidad en definir la arquitectura de la red, es decir, la red puede tener tantas capas y neuronas como queramos (o quepan en la memoria de nuestra computadora). También podemos elegir la función de activación para nuestras capas ocultas. Estas características arbitrarias de la red se llaman hiperparámetros, y se diferencian de los parámetros de la red (pesos y sesgos) porque los hiperparámetros quedan fijos al momento de definir la red, y no se modifican posteriormente.

En esta semana nos centraremos en el proceso de **entrenamiento** de una red neuronal. Es decir, como lograr que nuestra red aprenda a resolver una tarea partir de un conjunto de datos de ejemplo. Para ello, necesitaremos ajustar los parámetros de la red, es decir, modificar los valores numéricos de los pesos y sesgos de modo que obtengamos un resultado satisfactorio.

## Conjunto de datos de entrenamiento

El conjunto de datos que utilizaremos para entrenar nuestra red se llama **conjunto de datos de entrenamiento**. La Figura [](#) presenta una representación de un conjunto de datos $D$. Cada elemento del conjunto de datos $(x_i, y_i)$ está conformado por un conjunto de números, pero también por su respectiva etiqueta. 



:::{figure} ../_images/01_entrenamiento.png
:label: 01_entrenamiento
Conjunto de datos de entrenamiento con su respectiva etiqueta. Los datos corresponden a N imágenes de 28x28 pixeles que representan dígitos escritos a mano $x_i$ y su respectiva etiqueta $y_i$.
:::


:::{important}
**Nota**

Una imágen en escala de grises corresponde simplemente a un arreglo de números, que típicamente adquieren valores entre 0 y 255. 

Por ejemplo, considera el siguiente arreglo de números 

```
[  8,  12,  30, 200, 230, 210 ]
[ 10, 180, 240, 255, 220,  40 ] 
[ 15, 210, 255, 240,  80,  12 ]
[  5,  20, 200, 255, 210,  10 ] 
[  8, 100, 240, 230, 180,  15 ] 
[ 10,  15,  60, 190, 210,  20 ] 
```

Si pintamos cada pixel con una intensidad que depende del valor del píxel se logra la imágen de la [](#02_pixeles).
:::


:::{figure} ../_images/02_pixeles.png
:label: 02_pixeles
Representación de un arreglo de números como una imágen. El valor 0 es representado por el color negro, mientras que el color blanco representa el valor numérico 255. Los grises representan toda la gama intermendia de valores.
:::


 ## Diseñando la arquitectura de una red neuronal simple

Pongámonos a la tarea de crear una red que permita identificar el dígito (`0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8` ó `9`) a partir de las imágenes.

:::{tip}
**Pregunta clave**

¿cómo tiene que ser la arquitectura de la red para cumplir esta tarea?
:::

Lo primero que debemos pensar es en la capa de entrada. Sabemos que cada imágen es de 28x28 pixeles, es decir, en total tenemos 784 valores que servirán de entrada. Para mantener nuestro ejemplo sencillo, pensemos en una arquitectura que tenga 2 capas ocultas, cada una con 128 y 64 neuronas respectivamente, tal como se muestra en la [](#mnist_network_architecture).

Finalmente tenemos la capa de salida, la cual se tiene que ajustar a lo que queremos predecir. Como existen 10 clases posibles (una por cada dígito), nuestra capa de salida tiene que tener 10 neuronas. Nos gustaría que cada una de estas neuronas represente la probabilidad de cada una de las clases. Observe la capa de salida de la red en la Figura #.


:::{important}
**Importante**

Notar que debemos inicializar los parámetros de la red, es decir, definir cuales serán los valores numéricos iniciales de los pesos y sesgos de la red ¿con qué valores lo inicializarías?
:::

Más adelante veremos como llevar esta arquitectura a código, pero antes necesitamos entender algunos puntos claves de como funciona una red neuronal.

:::{figure} ../_images/mnist_network_architecture_v3.svg
:label: mnist_network_architecture
Representación de una red neuronal simple sin entrenar que aborda el problema de clasificación de imágenes de dígitos escritos a mano. Notar que no se han dibujado todas las conexiones para mantener la legibilidad del diagrama.
:::


## Propagación hacia adelante

El flujo de una red neuronal, o como la información de los datos de entrada se propaga a través de la red es la siguiente: 
1. primero los valores que están en la capa de entrada (pixeles de la imágen) pasan  multiplicandose por los pesos de las neuronas de la primera capa oculta. Luego se adiciona los segos y se aplica la función de activación.
2. Continuando con el flujo de la información, los 129 valores resultantes de la primera capa oculta pasan a la segunda capa oculta, multiplicandose por los respectivos pesos y sumando los sesgos. 
3. Finalmente los 64 valores que salen de la segunda capa oculta pasan a la capa de salida. El valor resultante de cada neurona de la capa de salida se interpreta como la probabilidad de que la imágen corresponda a una clase particular. Para ello se necesita utilizar una función de activación especial que se llama **softmax**, la cual tiene la particularidad que si se suman todos los valores de las salidas totalizan 1, o equivalentemente, 100%. 

A este flujo completo se le llama **propagación hacia adelante**, o en inglés  *forward pass*.



## ¿Cuánto se está equivocando la red?

Para responder esta pregunta necesitamos dos ingredientes esenciales: por una lado las etiquetas de los datos, los cuales nos indican la clase verdadera a la cual corresponde cada imágen.

El segundo ingrediente es "una medida" que nos permita saber que tan bien o mal lo está haciendo la red. Por ejemplo, en la figura vemos que la imágen se asemeja a un número 3, sin embargo la red le otorga la mayor probabilidad a la clase del dígito 0. Es decir, no lo está haciendo muy bien, pero **¿cuán mal lo está haciendo?** es imporante que cuantifiquemos ese error.

Para poder determinar si la red está calculando de manera correcta, necesitamos contar con alguna medida -un número- que permita calcular la diferencia entre lo esperado y lo calculado. Es decir, una medida que nos indique cuánto se está equivocando la red.

Este proceso se hace a través de la denominada **función de error**. La función de error $\mathcal{L}$ representa la diferencia entre la salida esperada $y^{(i)}$ para el ejemplo $i$, y la salida calculada $\hat y^{(i)}$ para ese mismo caso.


$$
\text{error} =\text{error}(y^{(i)}, \hat y^{(i)})
$$
Es importante señalar que el resultado de esta función de error es un número, y entre más cercano a cero, más pequeño es el error que está cometiendo la red. 

## Pérdida

El error se calcula para un dato en particular, sin embargo, esto no es suficiente. El conjunto de datos de entrenamiento puede incluir centenares de miles de ejemplos, por lo que lo que realmente necesitamos es un indicador para todos estos casos. Entonces, lo que podemos hacer es promediar el error de todos los datos. A este error promedio se le llama  **pérdida** o en inglés *loss*, y que simbolizaremos por la letra $\mathcal{L}$. Matemáticamente se define como:
$$
\mathcal{L} =\frac{1}{N}\sum_{i=1}^{N}\text{error}(y^{(i)}, \hat y^{(i)})
$$

:::{important}
**Importante**

El error promedio de la red depende únicame de sus parámetros $\theta$,  es decir, de los valores de los pesos y los sesgos ($W_i$ , $b_i$, $U$ y $c$). Si elegimos sabiamente los valores para los parámetros podemos reducir el error promedio.
:::

### Entropía Cruzada

En particular, para problemas de **clasificación**, utilizaremos como función de error la **entropía cruzada**. Esta función mide qué tan diferente es la distribución de probabilidades que predice la red respecto a la distribución real (la etiqueta verdadera). Para entenderla intuitivamente, imagina que la red tiene que "apostar" por una de las 10 clases: si la red le asigna una probabilidad alta a la clase correcta, el error será pequeño; pero si la red está muy segura de la respuesta incorrecta, el error será muy grande. Matemáticamente, para un ejemplo $i$, la entropía cruzada se define como:
$$
\text{error}(y^{(i)}, \hat{y}^{(i)}) = -\sum_{k=1}^{K} y_k^{(i)} \log\left(\hat{y}_k^{(i)}\right)
$$
donde $K $ es el número de clases (en nuestro caso, 10 dígitos), $y_k^{(i)} $ es 1 si la clase correcta es $k$ y 0 en caso contrario, y $\hat{y}_k^{(i)} $ es la probabilidad que la red le asigna a la clase $k$. En la práctica, como solo una clase es la correcta, la suma colapsa a un único término: $-\log(\hat{y}_{\text{correcta}}^{(i)}) $. Esto significa que el error crece rápidamente cuando la red le asigna una probabilidad baja a la respuesta correcta, lo que la "obliga" a aprender a ser más precisa.

:::{tip}
**Pregunta**

¿Cómo lograremos que nuestra red aprenda a resolver una tarea partir de un conjunto de datos de ejemplo?

Respuesta: Ajustando los parámetros de la red. Es decir, modificar los valores numéricos de los pesos y sesgos de modo que obtengamos un resultado satisfactorio.
:::
