---
title: Introducción a Redes Neuronales Convolucionales
subject: Aprendizaje Profundo
subtitle: Construyendo arquitecturas basadas en la naturaleza
short_title: Intro Redes Convolucionales
authors:
  - name: Jorge Anais
    orcid: 0000-0001-9051-1338
    email: jrganais@gmail.com
license: MIT
keywords: perceptron, funcion de activación
---

## Motivación

La historia de las Redes Neuronales Convolucionales (CNN) representa, quizás, el caso de éxito más fascinante de la inteligencia artificial inspirada en la biología. Durante las décadas de 1950 y 1960, los neurofisiólogos David Hubel y Torsten Wiesel estudiaron las bases del sistema visual de los mamíferos. Sus hallazgos no solo les valieron el Premio Nobel, sino que sentaron las bases para el diseño de las redes neuronales que hoy permiten a las computadoras "ver" (Hubel y Wiesel, 1959, 1962, 1968).

Hubel y Wiesel observaron cómo las neuronas en el cerebro de los gatos respondían a imágenes proyectadas en lugares específicos de su campo visual. Su mayor descubrimiento fue que las neuronas en las etapas tempranas del sistema visual reaccionaban intensamente a patrones de luz muy específicos, como barras orientadas en ángulos precisos, mientras que ignoraban otros patrones.

:::{figure} ../_images/cap2/gato.png
:label: gatoexperimento
Representación del montaje del experimento de Hubel y Wisel. El experimento realizado en las décadas de 1950 y 1960, demostró cómo la corteza visual cerebral procesa imágenes, identificando neuronas especializadas en detectar líneas, bordes y movimiento. Fuente: https://www.timetoast.com/timelines/a-shamelessly-awesome-history-of-modern-psychology
:::





Hoy sabemos que el sistema visual procesa la luz comenzando en la retina, desde donde la señal viaja a través del nervio óptico y el núcleo geniculado lateral hasta alcanzar la corteza visual primaria (V1), una región que preserva fielmente la organización espacial del campo visual original. Dentro de V1, el análisis inicial de la imagen recae en dos tipos de células especializadas: las primarias, que detectan características precisas como bordes y orientaciones en ubicaciones exactas, y las complejas, que identifican estos mismos patrones pero de forma independiente a su posición precisa. A medida que esta información procesada avanza hacia capas anatómicas más profundas, el sistema visual aplica iterativamente estos mismos principios de detección y agrupación, permitiendo que las neuronas reconozcan conceptos cada vez más abstractos y se vuelvan invariantes frente a transformaciones externas como cambios en la escala o la iluminación.

Este fenómeno dio lugar a la teoría de las **"neuronas de la abuela"**. Esta es la idea de que una persona posee neuronas que se activan específicamente al ver a su abuela, sin importar si está a la izquierda o derecha, si la imagen es un primer plano o de cuerpo completo, o si está en sombras.

Estos estudios sobre la corteza visual sirvieron de inspiración para la creación del neocognitrón (Fukushima, 1979), un modelo pionero que evolucionó gradualmente hasta convertirse en lo que hoy conocemos como Redes Neuronales Convolucionales (CNN).

Un hito fundamental en esta evolución fue la publicación de un artículo en 1998 por Yann LeCun y colaboradores. En este trabajo se presentó la famosa arquitectura LeNet-5, la cual marcó un antes y un después en la aplicación práctica de la inteligencia artificial, siendo adoptada masivamente por los bancos para reconocer dígitos escritos a mano en los cheques.

## Redes neuronales convolucionales

Con esta inspiración en la naturaleza de la visión, estamos listos para adentrarnos en el desarrollo de las Redes Neuronales Convolucionales (CNN). 

###  La convolución

El bloque más importante de una CNN son las capas convolucionales, y para poder definirlas necesitamos entender la operación de convolución en el procesamiento de imágenes.

La convolución discreta es una operación matemática que toma una entrada -típicamente una imágen- y un filtro o *kernel*. El resultado de esta operación se llama mapa de características. Matemáticamente se describe de la siugiente forma
$$
S(i, j) = (I * K)(i,j) = \sum_m \sum_n I(m,n) K(i-m, j-n)
$$
donde $I$ es una imágen (matriz de 2 dimensiones), $K$ es el kernel (matriz de 2 dimensiones), y los índices  $(m,n)$ e $(i, j)$ indican las coordenadas de la imágen y el kernel resepectivamente. Podemos pensar esta fórmula como un proceso iterativo, donde el kernel se va desplazando a través de la imágen y en cada iteración se realiza la multiplicación elemento a elemento y luego se suman estos valores. Esto se ilustar en la animación de la [](#padding_strides).  El resultado es un mapa de características $S$ que también es una matriz en 2 dimensiones.

En la animación de la [](#padding_strides) se ilustra la operación principal de una capa convolucional. El filtro o *kernel* (representado en la parte superior en color verde oscuro) es una matriz de $3 \times 3$ que se desliza sobre la imagen de entrada (representada en azul, de dimensión $5 \times 5$).

Durante la convolución, el filtro recorre la imagen de entrada partiendo desde la esquina superior izquierda. Para evitar la pérdida de información en los bordes y la reducción prematura de las dimensiones espaciales, se agregan píxeles artificiales (generalmente con valor cero) alrededor del perímetro de la imagen original. Estos píxeles están representados por las celdas con bordes punteados. A esta técnica se le conoce como **padding**.

El tamaño del salto que da el filtro en cada iteración se denomina zancada (*stride*). En este caso particular, en cada iteración el filtro avanza dando 2 pasos a la derecha y 2 pasos hacia abajo ($S=2$). Es importante conocer la relación entre estos parámetros al diseñar capas convolucionales, ya que determinan la dimensión del mapa de características resultante. Podemos calcular el tamaño de salida ($O$) mediante la siguiente fórmula:
$$
O = \lfloor \frac{W - K + 2P}{S} \rfloor + 1
$$
Donde $W$ (*width*) es la dimensión de la entrada ($5$), $K$ (*kernel*) es el tamaño del filtro ($3$), $P$ (*padding*) es la cantidad de borde agregado ($1$) y $S$ (*stride*) corresponde al tamaño del paso ($2$).


:::{figure} ../_images/cap2/padding_strides.gif
:label: padding_strides
Representación del calculo de la convolución mediante una animación. El cuadro azul representa la imágen de entrada, mientras que el cuadro verde representa el filtro o kernel. La animación muestra como el filtro se va desplazando a lo largo y ancho de la imágen de entrada. Fuente: https://hannibunny.github.io/mlbook/neuralnetworks/convolutionDemos.html
:::

:::{tip}
**Actividad**

Realiza la actividad propuesta en el cuadernillo interactivo `010_dim_conv.ipynb` para aplicar el cálculo de las dimensiones de salida en la operación de convolución.
:::

### ¿Por qué usar la convolución?

Mira la [](#katerinavulcova). ¿qué animal puedes identificar en la fotografía? Nuestro cerebro es muy bueno identificando patrones visuales y en una fracción de segundo sabemos que se trata de un gato.

:::{figure} ../_images/cap2/katerinavulcova-cat-9125207.jpg
:label: katerinavulcova
¿Cómo podemos reconocer que en la fotografía hay un gato? El reconocimiento del animal se logra mediante patrones visuales característicos: la disposición triangular de las orejas, la mirada frontal de los ojos de pupila vertical, y la estructura simétrica de la nariz y boca. Estos rasgos forman una configuración espacial única que permite identificar la categoría "gato" frente a otros objetos o animales, independientemente del entorno. Fuente: https://pixabay.com/es/photos/gato-gatito-mascota-gato-joven-9125207/ 
:::

Sin embargo, lograr que una máquina realice este reconocimiento a partir de una fotografía no es tan fácil. Las  claves que ha permitido los avances actuales en la visión por computadora, sobre todo utilizando redes neuronales, son:

- **Interacciones locales** Para identificar la nariz o un ojo del gato necesitamos observar píxeles que están próximos entre sí. No tiene sentido que la red intente conectar un píxel de la esquina inferior (pasto) con un píxel del centro (ojo) para reconocer su forma. Al concentrarnos en regiones pequeñas, la red se vuelve más eficiente y requiere menos memoria.
- **Compartición de rasgos:** Si aprendemos un filtro capaz de detectar el borde curvo de una oreja en la parte superior, ese mismo conocimiento es útil para detectar rasgos similares en otras partes. En lugar de aprender qué es una "oreja" para cada posición exacta de la foto, usamos el mismo filtro (kernel) en toda la imagen, permitiendo que el sistema sea más ligero y estadísticamente robusto.


Resumiendo, en vez de procesar la imagen de golpe, la red utiliza la convolución para construirla a partir de rasgos locales, imitando la visión humana, tal como se ilustra en la [](#GatoChollet).





:::{figure} ../_images/cap2/GatoChollet.png
:label: GatoChollet
Representación jerárquica de los elementos que comprenden una imágen de un gato según la analogía de como funciona nuestro sistema visual. El mundo visual forma una jerarquía espacial de módulos visuales: las líneas o texturas elementales se combinan en objetos simples como ojos u orejas, los cuales a su vez se combinan en conceptos de alto nivel como 'gato'. Fuente: François Chollet (2021). *Deep Learning with Python*.
:::




## Las  capas convolucionales

Una capa de una red típica no solo consta únicamente de la operación convolución, sino ocurre en tres etapas:

1. **Etapa de Convolución:** Se ejecutan varias convoluciones en paralelo. Cada una de estas operaciones se realiza con un kernel distinto, y se obtiene como resultado un mapa de características para cada xaso. Una ilustración de este proceso se muestra en la [](#convolucion).
2. **Etapa de Activación:** Se aplica una función de activacion (comúnmente **ReLU**) a la salida de la convolución.
3. **Etapa de Pooling:** Se aplica una operacion de pooling para reducir el tamaño de los mapas de características resultantes.



:::{figure} ../_images/cap2/convolucion.png
:label: convolucion
Esquema de una capa convolucional aplicada a una imagen RGB. Una imagen de entrada X con 3 canales $(R, G, B)$ es convolucionada con n kernels $(k_1, k_2,\dots, k_n)$. Cada kernel posee el mismo número de canales que la imagen de entrada (3 canales), de modo que cada convolución produce un único mapa de activación. Al aplicar los $n$ kernels se obtiene la salida $X\times K$, un volumen con $n$ canales, donde cada canal corresponde al mapa de características generado por un kernel distinto.
:::



### ¿Qué hace el Pooling?

El pooling (o submuestreo) en una red neuronal convolucional actúa como un mecanismo de resumen. Su función principal es reducir las dimensiones espaciales (ancho y alto) de los mapas de características que genera la convolución, conservando únicamente la información más importante.

¿Recuerdas las células complejas de la corteza visual primaria? El pooling es, en esencia, la operación matemática que imita esa función. Al resumir una región local de la imagen, le otorga a la red invarianza espacial. Esto significa que si un rasgo específico (como el borde de la oreja de un gato) se mueve ligeramente o cambia un poco de escala, la red lo seguirá detectando sin depender de su ubicación exacta.

Hay dos tipos principales de pooling:

- **Max Pooling:** Elige el valor máximo (el rasgo más fuerte) .
- **Average Pooling:** Calcula el promedio.

La [](#maxpool) muestra un ejemplo de pooling.


:::{figure} ../_images/cap2/maxpool.png
:label: maxpool
Operación de max pooling con filtro $2\times2$ y stride 2. Se aplica *max pooling* sobre un mapa de activación de $4\times4$. La ventana deslizante de $2\times2$ recorre la entrada con una zancada (*stride*) de 2, sin solapamiento entre regiones. De cada una de las 4 subregiones coloreadas se retiene únicamente el valor máximo, produciendo una salida de $2\times2$. Esta operación reduce la resolución espacial a la mitad en cada dimensión, disminuyendo el costo computacional y aportando invarianza local a pequeñas traslaciones.
:::



## Arquitectura típica de una red convolucional

Un arquitectura de una red convolucional tipicamente apila capas convolucionales, cada una seguida por una activación ReLU y una capa de pooling, como se ilustra en la [](#cnn_jorge). La imagen se hace cada vez más pequeña a medida que avanza por la red, pero también se vuelve típicamente más y más profunda, es decir, con más mapas de características, gracias a las capas convolucionales. En la parte superior de la pila, se añade una red neuronal *feedforward* normal, compuesta por unas pocas capas completamente conectadas, y la capa final produce la predicción. Por ejemplo, una capa softmax que emite las probabilidades estimadas de las clases.


:::{figure} ../_images/cap2/cnn_jorge.png
:label: cnn_jorge
Arquitectura típica de una red convolucional. Los recuadros verdes representan los kernes de las respectivas capas convolucionales, mientras que los azules corresponden a los mapas de caractarísticas. El diagrama ilustra cómo los datos de entrada atraviesan capas alternadas de convolución y *pooling*, reduciendo gradualmente sus dimensiones espaciales mientras aumentan su profundidad (mapas de características), para finalmente ser procesados por capas densas que generan la predicción de salida. 
:::





Así es como puedes implementar una red convolucional típica en PyTorch

```python
from functools import partial
DefaultConv2d = partial(nn.Conv2d, kernel_size=3, padding="same")
model = nn.Sequential(
    DefaultConv2d(in_channels=1, out_channels=64, kernel_size=7), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    DefaultConv2d(in_channels=64, out_channels=128), nn.ReLU(),
    DefaultConv2d(in_channels=128, out_channels=128), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    DefaultConv2d(in_channels=128, out_channels=256), nn.ReLU(),
    DefaultConv2d(in_channels=256, out_channels=256), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Flatten(),
    nn.Linear(in_features=2304, out_features=128), nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(in_features=128, out_features=64), nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(in_features=64, out_features=10),
).to(device)
```




:::{tip}
**Actividad**

Para profundizar como implementar una red neuronal convolucional en PyTorch revisa el notebook `EA2_S5_A4_CNN_MNIST.ipynb`.
:::



## Data augmentation

El aumento de datos (*data augmentation*) es una técnica de regularización que reduce el sobreajuste ampliando artificialmente el conjunto de entrenamiento mediante la generación de variantes realistas y aprendibles de las imágenes originales. Utilizando bibliotecas como `torchvision.transforms.v2`, se pueden combinar modificaciones como rotaciones, traslaciones, cambios de escala, ajustes de color y volteos horizontales, tal como se muestra en la [](#variaciones_de_cheems). Este proceso obliga al modelo a volverse más tolerante y robusto ante diferentes condiciones de iluminación y variaciones espaciales de los objetos, mejorando significativamente su capacidad de generalización.


:::{figure} ../_images/cap2/variaciones_de_cheems.png
:label: variaciones_de_cheems
Variaciones de la imágen mediante aumentación de datos aleatoria (*data augmentation*). A partir de una fotografía original, se generaron múltiples variantes mediante transformaciones aleatorias: volteo horizontal con probabilidad de 50%, rotación aleatoria entre -36° y +36° y  acercamiento/alejamiento por un factor entre -20% a +20%. Dado que cada variante es una aumentación independiente, el modelo nunca procesa dos veces el mismo encuadre, orientación ni escala, simulando así un conjunto de entrenamiento más amplio y diverso que contribuye a reducir el sobreajuste. Fuente: Modificado de https://www.instagram.com/balltze/.
:::



## **Lectura complementaria** 

Para revisar conceptos con una mirada práctica se recomienda el libro *Hands-on machine learning with Scikit-Learn and PyTorch* por Aurélien Géron ( https://learning-oreilly-com.webezproxy.duoc.cl/library/view/hands-on-machine-learning/9798341607972/ ). El capítulo 12 introduce los conceptos asociados a las redes convolucionales artificiales.

