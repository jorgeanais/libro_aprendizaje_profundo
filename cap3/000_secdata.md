---
title: Introducción a Redes Neuronales Recurrentes
subject: Aprendizaje Profundo
subtitle: Modelando Datos Secuenciales
short_title: Intro Redes Recurrentes
authors:
  - name: Jorge Anais
    orcid: 0000-0001-9051-1338
    email: jrganais@gmail.com
license: MIT
keywords: perceptron, funcion de activación
---

Hasta ahora, nos hemos enfocado principalmente en datos tabulares e imágenes. Este tipo de estructuras son muy comunes y hemos comprobado cómo las arquitecturas MLP (Perceptrón Multicapa) y CNN (Convolucionales) resuelven con éxito las tareas asociadas a ellas.

Sin embargo, existe una vasta familia de datos fundamentales en nuestra vida cotidiana que hemos dejado fuera: las secuencias. Estos son datos que poseen una dependencia temporal o un orden lógico entre sí. Ejemplos de datos secuenciales son los siguientes:

- **Audio**: Variaciones de presión en el aire en función del tiempo.
- **Video**: Una sucesión de fotogramas donde el orden es crítico para entender el movimiento.
- **Texto**: Una cadena de palabras donde el significado depende de lo que se dijo antes.

Sin embargo, a diferencia de las imágenes, que podemos redimensionar a un tamaño fijo, las secuencias suelen tener **largos variables**.

Pensemos en el análisis de sentimiento: un comentario sobre un producto puede ser tan breve como *"Me encantó"*, o una reseña detallada de tres párrafos criticando la calidad de construcción. Lo mismo ocurre en este ejemplo. Imaginemos que quisiéramos clasificar el canto de aves chilenas, tendríamos que procesar desde el trino corto y ascendente de un cachudito, pasando por las clásicas tres notas del chincol, hasta los complejos gorjeos de una golondrina chilena. Cada uno tiene una duración y un ritmo distinto. La [](#pajaritos) muestra Oscilogramas comparativos del canto de cuatro especies de aves chilenas.


:::{figure} ../_images/cap3/pajaritos.png
:label: pajaritos
Oscilogramas comparativos del canto de cuatro especies de aves chilenas. Representación de la amplitud del sonido en función del tiempo (segundos) para las vocalizaciones del Cachudito, Chercán, Chincol y Golondrina chilena. Se observa una marcada variación interespecífica en la estructura temporal y la duración de los cantos. Fotografías obtenidas de Aves de Chile (https://www.avesdechile.cl/). Audios obtenidos de Ladera Sur (https://www.instagram.com/laderasur/).
:::



Si intentáramos procesar estos datos con una red tradicional, nos encontraríamos con dos problemas principales: 

- En un MLP, la capa de entrada debe tener un número fijo de neuronas. Si las secuencias varían de tamaño, la red no sabría cómo recibirlas de forma nativa.
- Una solución parche sería ajustar la entrada al ejemplo más largo del dataset y rellenar el resto con ceros (*padding*). No obstante, esto dispara el número de parámetros innecesariamente y, lo más grave, **ignora la estructura secuencial**. Un MLP trata a cada entrada como independiente, perdiendo la noción de secuencia, precisamente, donde reside la información más valiosa.

Las redes neuronales que estudiaremos en esta experiencia de aprendizaje resuelven estos desafíos mediante una técnica elegante: la recurrencia. Veremos cómo estas arquitecturas mantienen una "memoria" de lo que ha ocurrido antes y cómo ciertas sofisticaciones modernas nos permiten extraer el máximo provecho de los datos secuenciales.'

:::{tip}
**Actividad**

El cuadernillo `EA3_S7_A1_motivacion_secuencias.ipynb` ilustra los problemas discutidos anteriormente. 
:::


:::{warning}
**Nota**

Es importante aclarar que, aunque utilizaremos ejemplos de Procesamiento de Lenguaje Natural (NLP) para ilustrar cómo las redes "entienden" el orden de las palabras, este curso no profundizará en las complejidades técnicas del procesamiento de texto como *embeddings* o *tokenización*. Sin embargo, estas analogías son fundamentales porque representan el uso más intuitivo de la recurrencia en la vida real.
:::

## Redes recurrentes 

Así como las redes convolucionales (CNN) están diseñadas para procesar cuadrículas de valores (como imágenes), las **Redes Neuronales Recurrentes (RNN)** son la familia de modelos especializada en el procesamiento de secuencias de valores $x^{(1)}, \dots, x^{(N)}$.

A diferencia de las redes tradicionales, las RNN pueden escalar para procesar secuencias mucho más largas de lo que sería práctico para otras arquitecturas, y tienen la capacidad de manejar datos de longitud variable.

Para facilitar el estudio de las RNN, definimos la secuencia como un conjunto de vectores $x^{(t)}$, donde el índice $t$ (paso de tiempo) va desde $1$ hasta $N$.

Desde el punto de vista de la arquitectura, las RNN extienden la idea de los grafos computacionales al incluir ciclos. Estos ciclos representan la influencia del valor actual de una variable sobre su propio valor en un paso de tiempo futuro. Al "desenrollar" estos ciclos, podemos visualizar la red como una estructura profunda que repite la misma operación en cada eslabón de la cadena de datos.

## Desenrollado de un grafo de computación

Para comprender cómo una red neuronal recurrente (RNN) procesa la información, es fundamental entender el concepto de desenrollado (*unfolding*). 

Para entenderlo de forma intuitiva, imagine que está leyendo la frase "El gato estaba sobre el tejado": en lugar de procesar todas las palabras simultáneamente, la RNN las lee una por una. En el primer paso, ingresa la palabra "El" y genera un estado oculto $h^{(1)}$ que resume lo visto hasta ese momento. En el segundo paso, toma "gato" junto con $h^{(1)}$  y produce un nuevo estado $h^{(2)}$  que ya incorpora el contexto de las dos primeras palabras. Este proceso se repite hasta llegar a "tejado", momento en el cual el estado final $h^{(5)}$  condensa la información de toda la secuencia. 

Existen dos formas de visualizar una RNN:

1. **Diagrama de Circuito:** Es una representación compacta que muestra un nodo con una flecha que vuelve a sí mismo. El símbolo de un cuadrado negro indica un **retardo de un paso de tiempo**. 
2. **Grafo Desenrollado:** Es una representación expandida donde cada paso de tiempo se dibuja como un nodo independiente. Esta vista muestra explícitamente el flujo de información y cómo se comparten los parámetros.




:::{figure} ../_images/cap3/unfoldCNN.png
:label: unfoldCNN
Representación de una red recurrente en fomatos circuito y desenrollado. Esta red recurrente procesa la información de la entrada **x** incorporándola al estado **h**, que se propaga hacia adelante a través del tiempo. (Izquierda) Diagrama de circuito. (Derecha) La misma red vista como un grafo computacional desplegado, donde cada nodo está ahora asociado a un instante de tiempo particular. Imagen modificada de Godfellow (2016).
:::


El proceso de desenrollado ofrece dos beneficios para el diseño de redes:

- **Tamaño de entrada constante:** Independientemente de si la secuencia tiene 5 o 500 elementos, el modelo siempre aprende la misma regla de transición de un estado a otro.
- **Compartición de parámetros:** Se utiliza la misma función $f$ con los **mismos parámetros** $\theta$ en cada paso. Esto permite que el modelo generalice de igual manera para secuencias cortas o largas.



## Tipos de arquitecturas recurrentes

Las redes neuronales recurrentes (RNN) son extremadamente versátiles debido a su capacidad para manejar diferentes estructuras de entrada y salida. Dependiendo del problema que busquemos resolver, podemos clasificar sus arquitecturas según el flujo de los datos:

**1. Muchos a Uno (*Many-to-One*)**

En esta arquitectura, la red procesa una secuencia completa de datos para generar una única predicción o categoría al final. Un ejemplo clásico son las **Redes tipo Elman** aplicadas al **análisis de sentimiento**: el modelo lee todas las palabras de una reseña (secuencia) y determina si el comentario es positivo o negativo (valor único).

**2. Muchos a Muchos (*Many-to-Many*)**

Aquí, el modelo recibe una secuencia y devuelve otra secuencia como respuesta. Existen dos variantes principales:

- **Sincrónica:** Cada elemento de entrada genera un elemento de salida de forma inmediata (como el etiquetado de palabras en una frase).
- **Asincrónica (Encoder-Decoder):** La red procesa toda la secuencia de entrada antes de empezar a generar la de salida. Es la base de la traducción automática, donde una frase en español se transforma en una frase en inglés.

**3. Uno a Muchos (*One-to-Many*)**

Este diseño toma un único dato de entrada y genera una secuencia completa de valores. El caso de uso más común es el etiquetado de imágenes (*image captioning*): la red recibe una imagen (un solo dato estático) y genera como salida una descripción textual, palabra por palabra (secuencia).

La [](#arquitecturas) muestra un ejemplo para cada una de las arquitecturas planteadas anteriormente.



:::{figure} ../_images/cap3/arquitecturas.png
:label: arquitecturas
Arquitecturas de redes neuronales recurrentes según la relación entre entradas y salidas.  Arquitecturas típicas de redes neuronales con ejemplos aplicados al procesamiento de texto. (Arriba) Arquitectura "muchos a uno" para análisis de sentimiento. (Centro) Arquitectura "muchos a muchos" en su variante asincrónica Encoder-Decoder para tarea de traducción. (Abajo) Arquitectura "uno a muchos" para una tarea de *captioning*. Elaboración propia.
:::



## Red de Elman

Una red de Elman, también conocida como red neuronal recurrente estándar (*vanilla RNN*), es una arquitectura diseñada para procesar una secuencia de datos y entregar una salida única.

La [](#arquitecturas) muestra el esquema desenrollado de esta red. En él, podemos observar el flujo de la información paso a paso:

- **Inicialización:** El proceso comienza combinando el primer elemento de la secuencia de entrada $x^{(1)}$ (multiplicado por la matriz de pesos $U$) con un estado oculto inicial $h^{(0)}$ (multiplicado por la matriz de pesos recurrentes $V$). Esto genera el primer estado oculto activo de la red, $h^{(1)}$.
- **Transición temporal:** Para el segundo paso, el nuevo estado oculto $h^{(2)}$ se obtiene aplicando la misma lógica: se combina la entrada actual $x^{(2)}$ (ponderada por $U$) con el estado oculto del paso anterior $h^{(1)}$ (ponderado por $V$).
- **Propagación:** Esta dinámica iterativa se repite sucesivamente a lo largo del tiempo hasta procesar el último elemento de la secuencia, $x^{(N)}$, produciendo así el estado oculto final $h^{(N)}$.
- **Predicción:** Finalmente, este último estado $h^{(N)}$ —que actúa como un resumen de toda la secuencia procesada— se pasa por una capa final (multiplicándose por los pesos de salida $O$) para generar la predicción única $\hat{y}$.



Esta ecuación explica cómo la red calcula cada estado oculto $h^{(i)}$ (los círculos centrales del diagrama). Básicamente, describe cómo la red actualiza su memoria en cada paso de tiempo $i$.
$$
h^{(i)} = F(x^{(i)}, h^{(i-1)})
= F(x^{(i)}U + h^{(i-1)}V+b)
$$
Finalmente, con el último estado oculto $h^{(N)}$ se calcula la salida $\hat y$. 
$$
\hat y = \text{softmax}(h^{(N)}O+c)
$$
Debido a que el modelo procesa una secuencia de longitud $N$ para emitir $1$ sola respuesta, esta arquitectura se clasifica como una red de tipo "muchos a uno".


:::{figure} ../_images/cap3/RNNcon1salida.png
:label: arquitecturas
Esquema de una red de Elman. El diagrama ilustra como la red procesa una secuencia de datos en cada instante y entrega una salida única. Por simplicidad se han omitido los sesgos de cada capa.
:::



## Arquitectura codificador-decodificador

La arquitectura Codificador-Decodificador (*Encoder-Decoder*) es una arquitectura de redes neuronales recurrentes, clasificado como un modelo de secuencia a secuencia (*Many-to-Many* asincrónico). Su propósito principal es mapear una secuencia de entrada a una secuencia de salida donde ambas pueden tener longitudes diferentes.

El uso más emblemático de esta arquitectura es la traducción automática, aunque también se utiliza para resumir textos o generar respuestas en chatbots.

Como se observa en la [](#rnn_unfold), el modelo se divide en dos grandes bloques el codificador y el decodificador.

El objetivo del **codificador** es "leer" y comprender la información de entrada.

- La red procesa la secuencia de entrada paso a paso, desde $x^{(1)}$ hasta $x^{(N)}$. En cada paso, actualiza su estado oculto utilizando los pesos $U_E$ y $V_E$.
- Al terminar de leer toda la secuencia, el codificador produce un estado oculto final, denotado como $h^{(N)}$. Este vector es crucial, ya que actúa como un "resumen" matemático que contiene todo el significado y contexto de la frase original.

El objetivo del **decodificador** es "escribir" la respuesta basándose en lo que entendió el codificador.

- Partiendo desdse el vector de contexto $h^{(N)}$ del codificador y lo utiliza como su estado inicial $g^{(1)}$. El proceso de generación arranca introduciendo un token especial de inicio de secuencia, representado como `<ini>`.
- A partir de ahí, la red genera un estado oculto $g^{(t)}$ y produce una predicción $\hat{y}^{(t)}$ en cada paso (utilizando sus propios pesos $U_D$ y $V_D$), hasta que el modelo predice un token especial de finalización `<fin>`, momento en el cual se detiene.

### Teacher Forcing y la diferencia entre Entrenamiento y Predicción

El diagrama del decodificador muestra dos caminos posibles para la entrada de cada paso temporal, lo que ilustra la diferencia fundamental entre cómo aprende la red y cómo opera en el mundo real.

**Fase de Entrenamiento**

Durante el entrenamiento, queremos que la red aprenda rápido y no se confunda por sus propios errores tempranos. Aquí se utiliza la técnica de **Teacher Forcing**:

- En lugar de usar la predicción que acaba de hacer la red $\hat{y}^{(1)}$ como entrada para el siguiente paso, le proporcionamos la **respuesta correcta real** obtenida del conjunto de datos de entrenamiento (denotada como $\tilde{y}^{(1)}$ y conectada con flechas continuas).
- Es como un profesor corrigiendo al alumno en cada palabra para que el siguiente intento no parta de un error previo.

**Fase de Predicción o Inferencia**

Cuando el modelo ya está entrenado y se pone a prueba con datos nuevos, no tenemos las respuestas correctas ($\tilde{y}$).

- En este escenario, el decodificador se ve obligado a utilizar su propia predicción anterior $\hat{y}^{(1)}$ como la entrada para el siguiente paso temporal.
- En la figura, este flujo autónomo está representado por las **flechas punteadas curvas** que bajan desde la salida $\hat{y}$ hacia la entrada del siguiente nodo temporal.



:::{figure} ../_images/cap3/rnn_unfold_with_outputs_2.svg
:label: rnn_unfold
Esquema desenrollado de la Arquitectura Codificador-Decodificador. El codificador (bloque verde) procesa iterativamente la secuencia de entrada para comprimir la información en un vector de contexto final $h^{(N)}$. Este vector sirve como estado inicial para el decodificador (bloque azul), el cual genera la secuencia de salida paso a paso.
:::


### Ventanas Deslizantes

Para entrenar una red recurrente con series temporales, no podemos entregarle toda la historia de una vez; necesitamos fragmentarla mediante la técnica de ventanas deslizantes (*sliding windows*).

Imagina que tienes una serie de datos de 10 días y decides usar una ventana de tamaño 3. La red tomará los días [1, 2, 3] para predecir el día 4. Luego, la ventana "se desliza" un paso hacia adelante y toma los días [2, 3, 4] para predecir el día 5, y así sucesivamente. Este proceso convierte una única serie larga en múltiples ejemplos de entrenamiento de largo fijo, permitiendo que la red aprenda la relación de dependencia entre un pasado reciente y el valor inmediatamente siguiente. 

:::{tip}
**Actividad**

En el cuadernillo `EA3_S7_A2_RNN_serie_temporal.ipynb` explorarás los datos reales del metro de Chicago y descubrirás como preparar los datos y entrenar una RNN para predecir la demanda de pasajeros.
:::


### **Apilando Recurrencia**

Hasta ahora hemos visualizado la RNN como una estructura que se despliega horizontalmente a través del tiempo. Sin embargo, al igual que en las redes MLP y CNN, podemos aumentar la capacidad de aprendizaje de nuestro modelo añadiendo profundidad vertical.

En una RNN profunda (*stacked RNN*), la salida del estado oculto de la primera capa no se utiliza solo para generar una predicción, sino que sirve como la secuencia de entrada para una segunda capa recurrente.

- **¿Por qué apilar capas?** Esto permite que la red extraiga características en distintos niveles de abstracción temporal. La primera capa puede captar variaciones locales (como fonemas en audio), mientras que las capas superiores pueden captar estructuras más complejas (como palabras o frases).
- **Implementación:** En frameworks como Keras o PyTorch, al apilar capas, es vital configurar la capa anterior para que devuelva la secuencia completa (`return_sequences=True`) y no solo el último estado, permitiendo que la capa superior tenga datos para procesar en cada paso de tiempo.



:::{tip}
**Actividad**

Predecir el futuro a partir del pasado es uno de los problemas más comunes en la industria. En el cuadernillo `EA3_S7_A3_RNN_Keras.ipynb` verás por qué las redes recurrentes son la herramienta natural para abordarlos, y cómo se implementan en Keras. 
:::



## El desafío de las secuencias largas y la degradación de la memoria

A pesar de la elegancia de las redes de Elman, estas enfrentan un obstáculo crítico cuando la secuencia es muy extensa (por ejemplo, un párrafo de 200 palabras o una señal de audio de varios segundos): **la pérdida de memoria a largo plazo**. 

- A medida que la red procesa nuevos elementos, la influencia de los primeros pasos de tiempo en el estado oculto actual tiende a desvanecerse.

- En secuencias largas, no toda la información es útil. Una red de Elman estándar intenta recordarlo todo por igual, lo que satura el estado oculto con "ruido" o datos intermedios que no aportan a la predicción final.

- A veces, para entender una palabra al principio de una frase, necesitamos contexto que aparece al final. Las redes que hemos visto son estrictamente unidireccionales (solo miran hacia atrás).

Para mitigar esto, una técnica utilizada son las **RNN Bidireccionales**. Estas redes procesan la secuencia en dos sentidos: una sub-red la lee de principio a fin y otra de fin a principio. Al combinar ambos estados, la red tiene una visión "panorámica" de cada punto de la secuencia.

### **¿Y si la red pudiera decidir qué olvidar?**

Aunque las redes bidireccionales y el apilamiento ayudan, el verdadero salto ocurre cuando modificamos la neurona misma. La próxima semana revisaremos las **celdas de memoria** (como LSTM y GRU), arquitecturas sofisticadas que incluyen "compuertas" para decidir qué información merece ser guardada, qué debe ser olvidada y qué debe ser transmitida al futuro.



- **Referencias** 

  - François Chollet (2021). Deep Learning with Python. MANNING [Libro]. https://www.manning.com/books/deep-learning-with-python-second-edition
  - Géron, A. (2025). Hands-on machine learning with Scikit-Learn and PyTorch. O'Reilly Media. [Libro] https://www.oreilly.com/library/view/hands-on-machine-learning/9798341607972/ 

  - Goodfellow, I., Bengio, Y., y Courville, A. (2016). Deep learning. MIT Press. [Libro] [http://www.deeplearningbook.org](http://www.deeplearningbook.org/). 

  - Scardapane, S. (2024). Alice’s adventures in a differentiable wonderland. [Libro] https://doi.org/10.48550/arXiv.2404.17625. 

  **Lecturas de la semana** 

  - Construyendo y entrenando redes neuronales MLP. Géron, A. (2025). Hands-on machine learning with Scikit-Learn and PyTorch. Capítulo 13: Processing Sequences Using RNNs and CNNs. 
  - François Chollet (2021). Deep Learning with Python. Capítulo 10: Deep learning for timeseries.
  - Goodfellow (2016). Deep learning. MIT Press. Capítulo 10 Sequence Modeling: Recurrent and Recursive Nets