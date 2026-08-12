---
title: Consideraciones importantes
subject: Aprendizaje Profundo
subtitle: Poniendo a punto una red neuronal
short_title: Consideraciones importantes
authors:
  - name: Jorge Anais
    orcid: 0000-0001-9051-1338
    email: jrganais@gmail.com
license: MIT
keywords: perceptron, funcion de activación
---

A continuación revisitaremos algunos conceptos fundamentales de aprendizaje automático, y que son importantes a la hora de entrenar modelos de aprendizaje profundo.

### Particionamiento del conjunto de datos

Al trabajar con el aprendizaje automático, es fundamental distinguir entre dos objetivos principales y separados: 

- la **selección del modelo**: estimar el rendimiento de distintos modelos para elegir el más adecuado

- la **evaluación del modelo**: calcular el error de predicción, o error de generalización, que tendrá el modelo final ya elegido al enfrentarse a información completamente nueva.

En situaciones donde disponemos de una gran cantidad de datos, la mejor estrategia para abordar ambos problemas es dividir aleatoriamente nuestro conjunto de datos en tres partes: un conjunto de entrenamiento, uno de validación y uno de prueba. Cada uno de estos subconjuntos cumple un rol específico e irremplazable. 

- El **conjunto de entrenamiento** se utiliza para ajustar o entrenar los modelos

- El **conjunto de validación** sirve para estimar el error de predicción y así poder seleccionar el mejor candidato

- El **conjunto de prueba** se emplea exclusivamente para evaluar el **error de generalización** definitivo del modelo final. 

Es de muy importante que este último conjunto de prueba se mantenga completamente aislado, como si estuviera guardado en una "bóveda", y se saque a la luz únicamente al concluir todo el análisis de datos. Si cometiéramos el error de usar el conjunto de prueba repetidamente para elegir qué modelo tiene el menor error, terminaríamos obteniendo una visión demasiado optimista que subestimaría, a veces de forma sustancial, el verdadero error que el modelo tendría en la vida real.

No existe una regla general estricta sobre cuántas observaciones deben ir en cada una de las tres partes, ya que esto depende en gran medida de la relación señal-ruido en los datos y del tamaño total de la muestra. Sin embargo, una distribución típica suele asignar el 50% de la información para el entrenamiento, dejando un 25% para la validación y el 25% restante para la prueba. 


:::{figure} ../_images/train_val_test_split_diagramv2.png
:label: train_val_test_split
Esquema del particionamiento del conjunto de datos en entrenamiento, validación y prueba usado para la selección de modelo y estimación del error de generalización.
:::

:::{important}
**Nota**

En el aprendizaje profundo (*deep learning*) típicamente se manejan del orden de millones de datos, las proporciones tradicionales de división pierden sentido. Lo ideal es destinar la inmensa mayoría de la información al entrenamiento, ya que reservar apenas un 1% para validación y prueba sigue ofreciendo un volumen de muestras más que suficiente para una evaluación rigurosa. Lo verdaderamente crucial con estos conjuntos reducidos no es el porcentaje, sino garantizar que sean totalmente representativos, abarquen la mayor diversidad de escenarios posibles y eviten cualquier desbalance de clases.
:::

### El problema del sobreajuste y el subajuste

En el desarrollo de cualquier modelo de aprendizaje automático, el subajuste y el sobreajuste son los dos grandes problemas a los que siempre deberemos enfrentarnos. Encontrar ese punto medio perfecto es el verdadero desafío del *deep learning*, y a continuación te presentamos las claves fundamentales para entender y dominar ambos conceptos.

El **sobreajuste** suele presentarse cuando el modelo es demasiado complejo en comparación con la cantidad de datos disponibles o cuando los datos contienen mucha información irrelevante. En lugar de aprender la regla general, la red neuronal empieza a memorizar el "ruido" y los patrones aleatorios de ese conjunto específico.

Para combatir el sobreajuste y lograr que el modelo generalice correctamente, puedes aplicar las siguientes estrategias:

- **Simplificar el modelo:** Utilizar una arquitectura con menos parámetros, reducir la cantidad de características en los datos de entrada o aplicar restricciones adicionales al modelo (técnicas de regularización).
- **Aumentar el conjunto de datos:** Recopilar y utilizar un volumen mayor de datos de entrenamiento para que el modelo tenga más ejemplos de donde aprender.
- **Limpiar los datos (reducir el ruido):** Mejorar la calidad de la información corrigiendo errores en las etiquetas y eliminando valores atípicos o anomalías (*outliers*).

Como es de esperar, el **subajuste** es exactamente el extremo opuesto al sobreajuste. Ocurre cuando el modelo diseñado es demasiado simple para capturar la verdadera estructura y complejidad de los datos. Por ejemplo, si intentáramos predecir algo con múltiples matices, como el precio del bitcoin, utilizando un modelo lineal básico, este inevitablemente sufriría de subajuste. La realidad es mucho más compleja que la herramienta que estamos utilizando para medirla; por lo tanto, sus predicciones serán muy imprecisas y fallarán incluso con los propios datos de entrenamiento.

Para corregir este problema y permitir que la red aprenda los patrones necesarios, las principales alternativas son:

- **Seleccionar un modelo con más capacidad:** Elegir una arquitectura más compleja que cuente con una mayor cantidad de parámetros.
- **Mejorar los datos de entrada**. Utilizar más datos o de mejor calidad.

### Optimizando la capacidad de una red neuronal

La Figura # muestra la pérdida en función de la capacidad de una red neuronal artificial. Decimos que una red tiene mayor capacidad cuando posee más parámetros o ha sido entrenada durante más épocas, entre otros factores.

La **pérdida de entrenamiento** es aquella que se mide sobre los mismos datos utilizados para entrenar el modelo. La **pérdida de generalización**, en cambio, se mide sobre datos que el modelo no ha visto durante el entrenamiento.

Del gráfico se pueden extraer las siguientes observaciones:

- La curva azul (pérdida de entrenamiento) desciende de forma monótona a medida que aumenta la capacidad de la red.
- La curva roja (pérdida de generalización) presenta una forma de U: disminuye hasta alcanzar un mínimo y luego comienza a crecer.

La interpretación es la siguiente. En la zona de subajuste, el modelo carece de la capacidad suficiente para capturar los patrones presentes en los datos. En la zona de sobreajuste, ocurre el efecto contrario: la capacidad del modelo es tal que comienza a memorizar los ejemplos de entrenamiento en lugar de aprender el patrón subyacente, perdiendo así su capacidad de generalizar a datos nuevos. El punto óptimo se encuentra donde la pérdida de generalización alcanza su valor mínimo, logrando el mejor equilibrio posible entre ambos extremos.




:::{figure} ../_images/bias_variance_tradeoff_es.svg
:label: bias_variance_tradeoff_es
Pérdida de entrenamiento y de generalización en función de la complejidad del modelo. Elaboración propia.
:::

:::{hint}
**Actividad 1**  

Analice y ejecute el cuaderno interactivo `error_generalizacion.ipynb`. El objetivo de este ejercicio es entrenar una red neuronal de dos capas ocultas para clasificar imágenes del conjunto de datos CIFAR-10, el cual contiene 60,000 imágenes distribuidas en 10 categorías de objetos.

Tras completar el entrenamiento en el notebook, responda de manera argumentada las siguientes preguntas:

1. ¿Qué relación observa entre la curva de pérdida de entrenamiento (*training loss*) y la pérdida de validación/generalización (*validation loss*)?
2. Basándose en las métricas obtenidas, ¿considera que la red presenta señales de sobreajuste (overfitting), subajuste (underfitting) o un ajuste óptimo?
3. ¿Qué cree que se podría hacer para mejorar la clasificación y la capacidad de generalización del modelo?
:::


### El balance sesgo-varianza

Un resultado teórico fundamental del aprendizaje automático establece que el error de generalización de un modelo puede expresarse como la suma de tres tipos de error cualitativamente distintos:

- **Sesgo** Esta componente del error de generalización se debe a suposiciones incorrectas sobre los datos, como asumir que su comportamiento es lineal cuando en realidad es cuadrático. Un modelo con alto sesgo tenderá a subajustarse a los datos de entrenamiento.

- **Varianza** Esta componente se debe a la excesiva sensibilidad del modelo ante pequeñas variaciones en los datos de entrenamiento. Un modelo con muchos grados de libertad (como un modelo polinomial de alto grado) tenderá a tener alta varianza y, por tanto, a sobreajustarse a los datos de entrenamiento.

- **Error irreducible** Esta componente se debe al ruido inherente a los propios datos. La única forma de reducirla es mediante la limpieza de los datos (por ejemplo, corrigiendo las fuentes de datos defectuosas, como sensores averiados, o detectando y eliminando valores atípicos).

Aumentar la complejidad de un modelo típicamente incrementa su varianza y reduce su sesgo. A la inversa, reducir la complejidad del modelo (o aumentar la regularización) incrementa el sesgo y reduce la varianza (véase la Figura #). De ahí que se hable de un balance o compromiso entre ambos términos.




:::{figure} ../_images/bias_variance_targets_es.svg
:label: bias_variance_targets_es
Representación del compromiso entre sesgo y varianza para modelos de aprendizaje automático. Modificado de Gerón (2025).
:::


## El problema del gradiente inestable

Durante el entrenamiento de una red neuronal, el algoritmo de retropropagación (*backpropagation*) viaja desde la capa de salida hacia la de entrada calculando los gradientes de error para actualizar los parámetros. Sin embargo, en redes profundas (redes con muchas capas), este proceso suele enfrentarse a graves problemas de inestabilidad en la velocidad de aprendizaje de las distintas capas:

- **Gradiente desvanecientes (Vanishing gradients):** A medida que el algoritmo avanza hacia las primeras capas (las más bajas), los gradientes se vuelven cada vez más pequeños. Como resultado, los pesos de esas conexiones apenas se actualizan y el entrenamiento nunca converge hacia una buena solución.
- **Gradiente explosivos (Exploding gradients):** Es el escenario opuesto, donde los gradientes crecen desproporcionadamente. Esto provoca actualizaciones de peso gigantescas que hacen que el algoritmo diverja, un problema muy común en las redes neuronales recurrentes.

A principios de los años 2000, este comportamiento errático provocó el abandono temporal de las redes neuronales profundas. No fue hasta 2010 que Xavier Glorot y Yoshua Bengio descubrieron a los principales culpables: la combinación de las técnicas de inicialización de pesos de la época y el uso generalizado de la **función de activación sigmoide**.

Como se puede observar en la gráfica  función sigmoide de la Figura # , cuando los valores de entrada se vuelven muy grandes (ya sean muy positivos o muy negativos), la función se "satura" alcanzando valores de 0 o 1. En estas zonas de saturación, la curva se vuelve plana, lo que significa que **su derivada es prácticamente cero**.

En consecuencia, cuando la retropropagación intenta hacer su trabajo desde las últimas capas saturadas, casi no hay gradiente que propagar hacia atrás. El poco gradiente que logra pasar se va diluyendo capa tras capa, dejando a las capas iniciales sin ninguna señal útil para poder aprender.


:::{figure} ../_images/sigmoid_plot.png
:label: sigmoid_plot
Gráfico de la función sigmoide. Se destacan la zonas de saturación para valores $z>4$ y $z < -4$. Para la región cercana a $z = 0$, el comportamiento es lineal. Figura modificada de Gerón (2025). 
:::


### La Importancia de la Inicialización de Parámetros

Por lo general, los parámetros de una red neuronal (pesos y sesgos) deben inicializarse con valores pequeños, cercanos a cero. Este paso es crucial, ya que le indica al algoritmo de descenso de gradiente su punto de partida exacto para comenzar a iterar y encontrar los valores que minimicen la función de error.

La semana pasada, en la red neuronal que construimos desde cero, inicializamos estos pesos de forma completamente aleatoria. Sin embargo, en arquitecturas más profundas, esto puede provocar el problema del desvanecimiento del gradiente. Para mitigarlo, se han desarrollado diversas heurísticas de inicialización:

- **Inicialización de Glorot (o Xavier):** Está diseñada para mantener un flujo de señal estable a través de las capas al igualar las varianzas de entrada y salida. Es ideal para redes que utilizan funciones de activación como la sigmoide.
- **Inicialización de He (o Kaiming):** Está optimizada específicamente para el uso de funciones de activación de tipo ReLU y sus derivadas.

Estas técnicas han sido fundamentales para acelerar los tiempos de entrenamiento y han sentado las bases del éxito del aprendizaje profundo (*deep learning*) moderno. 

Nota: Por defecto, la mayoría de las capas de PyTorch realizan una inicialización automática de los parámetros en su método `__init__` basado en estas heurísticas. 

:::{hint}
**Actividad 2**

Entra a la documentación de la capa  `nn.Linear` (https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) e indica que tipo de inicialización utiliza por defecto para los pesos y los sesgos.
:::

### Evolución de las Funciones de Activación

Además de una correcta inicialización de los pesos, la elección de la función de activación es clave para evitar la saturación de la red. Para abordar este problema, se propusieron opciones clásicas que hoy son muy populares, destacando **ReLU**, **Leaky ReLU**, **Parametric ReLU (PReLU)**, **ELU** y **SELU**. Se han graficado algunas funciones de activación en la Figura #.

Más recientemente, han surgido alternativas más complejas y de alto rendimiento, diseñadas a la medida de arquitecturas avanzadas. Entre ellas se encuentran **GELU**, **Swish**, **SwiGLU**, **Mish** y **ReLU²**.


:::{figure} ../_images/fnactivacion.png
:label: fnactivacion
Gráfica de algunas funciones de activación.
:::


En conclusión, aunque ReLU sigue siendo un excelente estándar por defecto gracias a su gran eficiencia computacional, las variantes modernas como Swish o Mish pueden ofrecer un rendimiento superior al enfrentarse a tareas de mayor complejidad.

:::{hint}
**Actividad 3**

Entra al siguiente enlace https://docs.pytorch.org/docs/stable/nn.html y revisa que funciones de activación están disponibles y completa el notebook `funciones_de_activacion.ipynb`.
:::

### Normalización por lotes (*Batch normalization*)

Si bien la inicialización de parámetros de Kaiming junto con la función de activación ReLU (o cualquiera de sus variantes) puede reducir considerablemente el riesgo de los problemas de desvanecimiento y explosión de gradientes al inicio del entrenamiento, no garantiza que estos no reaparezcan durante el proceso. 

En un artículo publicado en 2015, Sergey Ioffe y Christian Szegedy propusieron una técnica denominada normalización por lotes (*batch normalization*, BN) que aborda precisamente estos problemas. La técnica consiste en añadir una operación en el modelo inmediatamente antes o después de la función de activación de cada capa oculta. Dicha operación centra en cero y normaliza cada entrada. 

## Cómo evitar el sobreajuste mediante la regularización

Las Técnicas de Regularización son un conjunto de mejores prácticas que impiden activamente la capacidad del modelo para ajustarse perfectamente a los datos de entrenamiento, con el objetivo de hacer que el modelo funcione mejor durante la validación. 

La Regularización permite reducir la diferencia entre el error de entrenamiento y el error de generalización, como vimos en la Figura #.

### Regularización L1 y L2

Al igual  que en Machine Learning para los modelos lineales simples, es posible utilizar regularización L2 para restringir los pesos de conexión de una red neuronal, y/o la regularización L1 si se desea un modelo disperso (con muchos pesos iguales a cero). Como se mencionó anteriormente al discutir el optimizador AdamW, la regularización L2 es matemáticamente equivalente a la decaída de pesos (weight decay) cuando se utiliza un optimizador SGD (con o sin momento), por lo que en ese caso puede implementarse simplemente definiendo el argumento weight_decay del optimizador. A continuación se muestra cómo aplicar regularización L2 a los pesos de conexión de un modelo PyTorch entrenado con SGD, utilizando un factor de regularización de $10^{-4}$.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-4)
[...] # use the optimizer normally during training
```



### Drop Out

El dropout fue propuesto por Geoffrey Hinton y colaboradores en 2012 y posteriormente detallado en un artículo de 2014 por Nitish Srivastava y colaboradores. Desde entonces ha demostrado ser una técnica sumamente eficaz: muchas redes neuronales de vanguardia lo incorporan, logrando mejoras de precisión de entre un 1% y un 2%. Aunque pueda parecer marginal, cuando un modelo ya alcanza el 95% de precisión, una mejora del 2% implica reducir la tasa de error en casi un 40% (pasando de un 5% de error a aproximadamente un 3%).
El algoritmo es bastante sencillo: en cada paso del entrenamiento, cada neurona (incluidas las neuronas de entrada, pero excluyendo siempre las de salida) tiene una probabilidad p de ser temporalmente "desactivada", lo que significa que será ignorada por completo durante ese paso, aunque puede volver a estar activa en el siguiente. El hiperparámetro p se denomina tasa de dropout y típicamente se fija entre el 10% y el 50%: valores cercanos al  cercanos al 40%–50% en redes neuronales convolucionales (lo veremos en la unidad de aprendizaje 2) y del 20%–30% en redes neuronales recurrentes (lo veremos en la unidad de aprendizaje 3). Una vez concluido el entrenamiento, las neuronas dejan de desactivarse. 

![](https://storage.googleapis.com/kaggle-media/learn/images/a86utxY.gif)

Figura #. Representación de la técnica de regularización Dropout con un valor de p=50% entre ambas capas ocultas. Extraído de https://www.kaggle.com/code/ryanholbrook/dropout-and-batch-normalization/tutorial

### Detención temprana (*Early Stopping*)

Una forma alternativa de regularizar consiste en detener el entrenamiento en el momento en que el error de validación alcanza su mínimo. Esta popular técnica se denomina parada temprana (*early stopping*). La Figura# muestra un modelo complejo entrenado con descenso de gradiente por lotes sobre el conjunto de datos cuadrático utilizado anteriormente. A medida que avanzan las épocas, el algoritmo aprende y el error de predicción (RMSE) sobre el conjunto de entrenamiento disminuye, al igual que el error sobre el conjunto de validación. Sin embargo, tras un cierto número de épocas, el error de validación deja de decrecer y comienza a aumentar, lo que indica que el modelo ha empezado a sobreajustarse a los datos de entrenamiento. Con la parada temprana, simplemente se interrumpe el entrenamiento en cuanto el error de validación alcanza su mínimo. Es una técnica de regularización tan simple y eficaz que Geoffrey Hinton la llamó un "almuerzo gratis perfecto". 



## Optimizadores

Entrenar redes neuronales, especialmente aquellas muy profundas o con muchas capas, puede ser un proceso lento. Además de las estrategias ya vistas: buena inicialización de pesos, funciones de activación adecuadas, normalización por lotes o por capas, el uso de optimizadores más eficientes que el descenso de gradiente estándar representa una mejora sustancial en la velocidad de entrenamiento. 

**SGD** (*Stochastic Gradient Descent*): Es el método que hemos estado usando hasta ahora. Actualiza los parámetros utilizando subconjuntos de datos (mini-batches), lo que introduce ruido beneficioso para escapar de mínimos locales, aunque puede ser ineficiente en superficies de error complejas.

**SGD con Momentum**: Introduce una "memoria" de los gradientes previos. Al acumular el impulso de direcciones pasadas, ayuda a navegar por valles estrechos y acelera el avance hacia el mínimo global, reduciendo las oscilaciones.

**AdaGrad** (*Adaptive Gradient Algorithm*): Adapta la tasa de aprendizaje individualmente para cada parámetro. Es especialmente efectivo en datos dispersos (sparse data), ya que escala el aprendizaje en función de la frecuencia de actualización de cada característica.

**RMSProp (Root Mean Square Propagation):** Diseñado para resolver la caída prematura de la tasa de aprendizaje en AdaGrad. Utiliza un promedio móvil del cuadrado de los gradientes, lo que lo hace ideal para entornos no estacionarios y problemas de aprendizaje profundo recurrentes.

**Adam** (*Adaptive Moment Estimation*): Actualmente el estándar de la industria. Combina las ventajas del Momentum y de RMSProp, manteniendo estimaciones de los dos primeros momentos del gradiente. Es altamente robusto y requiere poca afinación de hiperparámetros.

**AdamW**: Una evolución de Adam que desacopla el decaimiento de los pesos (*weight decay*) del paso de actualización del gradiente. Esta modificación mejora significativamente la regularización y la capacidad de generalización, evitando el sobreajuste de forma más eficiente.

| **Optimizador**    | **Fortalezas**                                         | **Uso Ideal**                                     |
| ------------------ | ------------------------------------------------------ | ------------------------------------------------- |
| **SGD + Momentum** | Tiene una alta estabilidad y convergencia suave.       | Modelos de visión por computadora clásicos.       |
| **RMSProp**        | Excelente en caso de gradientes fluctuantes.           | Redes Neuronales Recurrentes (RNN).               |
| **Adam**           | Es eficiente computacionalmente y rápido.              | Propósito general y modelos complejos.            |
| **AdamW**          | Permite mejorar la generalización (menos sobreajuste). | Entrenamiento de Transformers y modelos modernos. |

:::{hint}
**Actividad 4**

Realiza el notebook `Optimizadores.ipynb`, en el se comparara el comportamiento de distintos optimizadores.
:::

:::{hint}
**Actividad 5**

Realiza el notebook `Ejercicio.ipynb`, en el se aplican regularización para mejorar el rendimiento de una red neuronal profunda de 20 capas. 
:::

:::{hint}
**Actividad 6**

Realiza el cuadernillo `red_con_keras.ipynb` donde revisaran un **Keras**, un framework de alto nivel, aplicado a un problema de clasificación con regularización.
:::



## **Lectura complementaria** 

Para revisar conceptos con una mirada práctica se recomienda el libro *Hands-on machine learning with Scikit-Learn and PyTorch* por Aurélien Géron ( https://learning-oreilly-com.webezproxy.duoc.cl/library/view/hands-on-machine-learning/9798341607972/ ). El capítulo 11 introduce los conceptos asociados al entrenamiento de redes neuronales artificiales aplicado con PyTorch.



## Revisemos lo aprendido

Analiza el caso presentado y elige la alternativa que mejor refleje lo aprendido en esta sección: 

:::{tip}
**Pregunta**

Una startup está desarrollando una aplicación móvil para identificar flores chilenas en tiempo real. Utilizan una red neuronal profunda con 30 capas. Tras varias horas de entrenamiento en PyTorch, el equipo observa el siguiente comportamiento en sus gráficas de rendimiento: 
- La pérdida de entrenamiento (training loss) es extremadamente baja, casi cercana a cero.
- La pérdida de validación (validation loss) comenzó a subir drásticamente después de la época 10.
- Al probar el modelo con fotos nuevas tomadas con un celular, el sistema confunde constantemente especies básicas.

El ingeniero líder sospecha que el modelo está "memorizando" el ruido de las imágenes de entrenamiento (como la iluminación o el fondo) en lugar de aprender las características de los pétalos.

**Dadas las señales observadas en el caso, ¿cuál es el diagnóstico más probable del estado del modelo y qué combinación de técnicas sería la más adecuada para resolverlo?**

A) El modelo sufre de **subajuste (underfitting)**; se debe aumentar la complejidad de la red y eliminar las capas de *Dropout*.  
B) El modelo presenta **gradientes explosivos**; la solución definitiva es cambiar todas las funciones de activación a tipo *Sigmoide*.  
C) El modelo sufre de **sobreajuste (overfitting)**; se recomienda aplicar *Early Stopping*, agregar *Dropout* y posiblemente aumentar el *weight decay* (regularización L2).  
D) El modelo tiene un **sesgo alto** y baja varianza; se debe dejar de usar el conjunto de validación y pasar directamente al conjunto de prueba ("bóveda").  
E) El problema es el **error irreducible** de los sensores del celular; ninguna técnica de regularización o cambio de optimizador puede mejorar los resultados.  

**Respuesta**: La brecha entre una pérdida de entrenamiento baja y una pérdida de validación que sube es el síntoma clásico del sobreajuste (el modelo pierde capacidad de generalización). Las técnicas mencionadas (Early Stopping, Dropout y L2) son las herramientas estándar discutidas en el texto para obligar al modelo a aprender patrones generales en lugar de memorizar el ruido. Por lo tanto, la respuesta correcta es la C.
:::