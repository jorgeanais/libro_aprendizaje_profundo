---
title: Profundizando sobre arquitecturas convolucionales
subject: Aprendizaje Profundo
subtitle: Arquitecturas notables
short_title: Arquitecturas Notables
authors:
  - name: Jorge Anais
    orcid: 0000-0001-9051-1338
    email: jrganais@gmail.com
license: MIT
keywords: perceptron, funcion de activación
---

A lo largo de los años, se han desarrollado diversas variantes de la arquitectura fundamental de las Redes Neuronales Convolucionales (CNN), lo que ha llevado a avances sorprendentes en el campo de la visión por computadora. Una excelente forma de medir este progreso es observar la evolución de las tasas de error en competencias como el desafío The ImageNet Large Scale Visual Recognition Challenge (en adelante ImageNet). En este certamen, la tasa de error para la clasificación de imágenes cayó de más del 26% en 2011 a menos del 3% en 2016.

Específicamente, esta métrica correspondía a la tasa de error *top-five*, es decir, la proporción de imágenes de prueba en las que la respuesta correcta no se encontraba entre las cinco predicciones de mayor confianza del sistema. Considerando que el conjunto de datos de ImageNet contiene imágenes grandes y 1,000 clases distintas (algunas tan sutiles que implican distinguir entre 120 razas de perros, ver Figura S6-01), analizar la evolución de las arquitecturas ganadoras es la mejor manera de entender cómo funcionan las CNN y cómo progresa la investigación en el aprendizaje profundo.



:::{figure} ../_images/cap2/image-18.png
:label: image-18
Imágenes de concurso ImageNet. Una muestra de imágenes del concurso ImageNet con sus respectivas categorías. Jia Deng y colaboradores (2008) https://www.image-net.org/static_files/papers/imagenet_cvpr09.pdf.
:::





## Recordando arquitecturas típicas de CNN

Como vimos las semana pasada, la arquitectura clásica de una CNN suele apilar unas cuantas capas convolucionales (generalmente seguidas de una capa de activación ReLU), luego una capa de agrupación (*pooling*), seguida de otras capas convolucionales más función de activación ReLU, otra capa de *pooling*, y así sucesivamente.

A medida que la imagen avanza por la red, sus dimensiones espaciales se vuelven cada vez más pequeñas, pero al mismo tiempo la red se vuelve típicamente más y más profunda (es decir, con una mayor cantidad de mapas de características) gracias a la acción de los filtros convolucionales. En la parte final de esta estructura, se añade una red neuronal densa (*feedforward*) tradicional compuesta por capas completamente conectadas, donde la capa final emite la predicción (por ejemplo, usando una función *softmax* para obtener las probabilidades estimadas por clase).

## La evolución de las arquitecturas

### AlexNet (2012)

La arquitectura AlexNet, desarrollada por Alex Krizhevsky, Ilya Sutskever y Geoffrey Hinton, ganó el desafío ImageNET de 2012 por un margen abrumador: logró un error *top-five* del 17%, mientras que el segundo lugar solo alcanzó el 26%. Fue la primera arquitectura en apilar múltiples capas convolucionales directamente una sobre otra, en lugar de intercalarlas siempre con capas de *pooling*. La siguiente tabla detalla la arquitectura utilizada capa a capa.

| **Capa** | **Tipo**        | **MdC** | **Tamaño** | **Kernel** | **Stride** | **Padding** | **Activación** |
| -------- | --------------- | ------- | ---------- | ---------- | ---------- | ----------- | -------------- |
| Out      | Fully connected | —       | 1,000      | —          | —          | —           | Softmax        |
| F10      | Fully connected | —       | 4,096      | —          | —          | —           | ReLU           |
| F9       | Fully connected | —       | 4,096      | —          | —          | —           | ReLU           |
| S8       | Max pooling     | 256     | 6 × 6      | 3 × 3      | 2          | valid       | —              |
| C7       | Convolution     | 256     | 13 × 13    | 3 × 3      | 1          | same        | ReLU           |
| C6       | Convolution     | 384     | 13 × 13    | 3 × 3      | 1          | same        | ReLU           |
| C5       | Convolution     | 384     | 13 × 13    | 3 × 3      | 1          | same        | ReLU           |
| S4       | Max pooling     | 256     | 13 × 13    | 3 × 3      | 2          | valid       | —              |
| C3       | Convolution     | 256     | 27 × 27    | 5 × 5      | 1          | same        | ReLU           |
| S2       | Max pooling     | 96      | 27 × 27    | 3 × 3      | 2          | valid       | —              |
| C1       | Convolution     | 96      | 55 × 55    | 11 × 11    | 4          | valid       | ReLU           |
| In       | Input           | 3 (RGB) | 227 × 227  | —          | —          | —           | —              |

Para combatir el sobreajuste (*overfitting*), sus autores utilizaron dos técnicas principales:

- **Dropout:** Aplicado con una tasa del 50% durante el entrenamiento en las salidas de las capas completamente conectadas superiores.
- **Data Augmentation:** Realizando desplazamientos aleatorios, volteos horizontales y cambios en las condiciones de iluminación de las imágenes originales.

AlexNet también introdujo una técnica llamada Normalización de Respuesta Local (**LRN**, por sus siglas en inglés), donde las neuronas más activadas inhiben a otras situadas en la misma posición en mapas de características vecinos. Esto fomenta que los mapas de características se especialicen, explorando una gama más amplia de patrones.

### GoogLeNet (2014)

Desarrollada por investigadores de Google (Christian Szegedy y colaboradores), ganó el desafío de 2014 empujando la tasa de error por debajo del 7%. Su rendimiento se debió a que era mucho más profunda que las redes anteriores, pero utilizando los parámetros de manera mucho más eficiente: GoogLeNet tiene unas 10 veces menos parámetros que AlexNet (aproximadamente 6 millones frente a 60 millones).

Esto fue posible gracias a la introducción de los **Módulos Inception**. Estos submódulos alimentan la señal de entrada a cuatro capas diferentes en paralelo (usando kernels de $1 \times 1$, $3 \times 3$ y $5 \times 5$) y concatenan sus salidas. De alguna manera, esto permite que la red pueda elijir por sí misma el tamaño del kernel adecuado en cada submódulo (o una combinación de estos).

:::{figure} ../_images/cap2/googlenet.png
:label: googlenet
Módulo Inception. La entrada se divide en cuatro ramas paralelas: una convolución 1×1 directa, una convolución 3×3 precedida por una reducción 1×1, una convolución 5×5 también con reducción 1×1, y un máx. pool 3×3 seguido de una convolución 1×1. La notación "3×3 + 1(S)" indica kernel 3×3, stride 1 y padding "same". Todas las capas convolucionales usan activación ReLU. Gracias al padding "same" en todas las ramas, los mapas de características mantienen las mismas dimensiones espaciales que la entrada, lo que permite concatenarlos a lo largo de la dimensión de profundidad en la capa final. 
:::




El uso de convoluciones de $1 \times 1$ cumple tres propósitos vitales:

1. Capturar patrones a lo largo de la dimensión de profundidad (entre canales).
2. Actuar como capas de **cuello de botella** (*bottleneck*), reduciendo la dimensionalidad computacional y el número de parámetros.
3. Funcionar en conjunto con otras capas convolucionales como extractores capaces de capturar patrones mucho más complejos a distintas escalas.

GoogLeNet también eliminó las capas completamente conectadas al final de la red, reemplazándolas por una capa de **Global Average Pooling**, la cual simplemente calcula el promedio de cada mapa de características, mitigando enormemente el riesgo de sobreajuste.}

Otro aspecto muy interesante de la arquitectura de GoogLeNet es que tiene salidas intermedias de predicción (ver octágonos blanco en la Figura ) que consisten en pequeñas redes neuronales que realizan una clasificación preliminar antes de llegar al final de la red. Su propósito principal es combatir el problema del gradiente desvanecido. Al añadir estas pérdidas intermedias durante el entrenamiento, se ayuda a propagar el gradiente hacia atrás de manera más efectiva, permitiendo que las capas iniciales aprendan mejor.

:::{figure} ../_images/cap2/image-20260417224656949.png
:label: googlenet
Arquitectura de GoogLeNet. El esquema representa una vista esquemática de la red GoogLeNet. Extraída de Szegedy y colaboradores (2015). https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html
:::




### ResNet (2015)

Kaiming He y su equipo arrasaron en el ILSVRC 2015 con una Red Residual (**ResNet**) que entregó un asombroso error menor al 3.6%. Su modelo ganador consistió en una red extremadamente profunda de 152 capas.

Entrenar redes tan profundas desde cero solía ser imposible debido a problemas matemáticos en el cálculo del error. La clave de ResNet fue introducir **conexiones de salto** (*skip connections* o *shortcut connections*), de esta forma la señal original de entrada de una capa se suma a la salida de una capa ubicada más adelante en la pila.

En lugar de forzar a la red a aprender una función completa $h(x)$, la conexión de salto obliga a la red a modelar una función residual:

$$f(x) = h(x) - x$$

Al inicio del entrenamiento, los pesos de una red son cercanos a cero. Gracias a estas conexiones, la red actúa como una función de identidad (la salida es igual a la entrada), lo que acelera dramáticamente el proceso. Además, permite que la señal y los gradientes fluyan limpiamente por toda la red, incluso si algunas capas tardan más en aprender. Las versiones más profundas (como ResNet-152) optimizan aún más esto utilizando unidades residuales que incluyen cuellos de botella de $1 \times 1$.


:::{figure} ../_images/cap2/conexionresidual.png
:label: conexionresidual
Bloque residual. A la izquierda, una red estándar donde las capas aprenden directamente la función objetivo *h*(**x**). A la derecha, un bloque residual: al sumar la entrada **x** a la salida de las capas (mediante una conexión directa), la red se ve forzada a aprender únicamente la función residual *f*(**x**) = *h*(**x**) − **x**, es decir, la diferencia entre la salida deseada y la entrada original. Este principio, conocido como *aprendizaje residual*, facilita el entrenamiento de redes muy profundas al permitir que el gradiente fluya directamente hacia las capas anteriores. 
:::


Posteriormente surgieron modelos como Inception-v4, que combinan eficientemente las ideas de módulos Inception con las conexiones residuales de ResNet.



## Transfer Learning: Aprovechando modelos preentrenados

Si deseas construir un clasificador de imágenes pero no tienes suficientes datos para entrenarlo desde cero, la mejor estrategia es usar el Aprendizaje Transferido (*Transfer Learning*).

Esta técnica consiste en reutilizar las capas inferiores de un modelo preentrenado (como las arquitecturas mencionadas anteriormente, previamente ajustadas en los millones de imágenes de ImageNet). Estas capas ya saben cómo detectar bordes, texturas, formas y conceptos complejos. Al reemplazar únicamente la "cabeza" de clasificación final de la red y realizar un ajuste fino (*fine-tuning*) con tu nuevo conjunto de datos, puedes obtener resultados de vanguardia.

Por ejemplo, si utilizas un modelo robusto preentrenado para clasificar un conjunto de datos pequeño como el *Flowers102* (que solo tiene 10 imágenes por clase), puedes superar fácilmente el 90% de precisión, algo imposible si intentaras entrenar la red desde cero.

**Actividad**

Para profundizar en la implementación práctica, abre el cuadernillo interactivo `EA2_S6_A1_TransferLearning.ipynb`. Allí reemplazarás la capa final de un modelo preentrenado en PyTorch y lo ajustarás para clasificar un conjunto de datos con pocas muestras.



## Implementación de una CNN ResNet-34 con PyTorch

La mayoría de las arquitecturas de redes neuronales convolucionales (CNN) pueden implementarse de forma muy natural en PyTorch. Aunque en la práctica profesional solemos cargar redes preentrenadas (transfer learning), entender cómo construir una **ResNet-34** desde cero es fundamental para dominar la flexibilidad del framework.

### 1. La Unidad Residual (`ResidualUnit`)

El corazón de una ResNet es su bloque residual. La idea principal es permitir que la red aprenda funciones de identidad mediante conexiones de salto (*skip connections*).


```python
class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Definimos una convolución estándar para evitar repetición de código
        DefaultConv2d = partial(
            nn.Conv2d, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Camino principal (derecha del diagrama)
        self.main_layers = nn.Sequential(
            DefaultConv2d(in_channels, out_channels, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            DefaultConv2d(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
        )
        
        # Conexión de salto (izquierda del diagrama)
        # Si hay un cambio en la dimensión (stride > 1), ajustamos la entrada
        if stride > 1:
            self.skip_connection = nn.Sequential(
                DefaultConv2d(in_channels, out_channels, kernel_size=1,
                              stride=stride, padding=0),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip_connection = nn.Identity()

    def forward(self, inputs):
        # Sumamos la salida del camino principal con la conexión de salto
        return F.relu(self.main_layers(inputs) + self.skip_connection(inputs))
```

Puntos clave del código:

- **Camino Principal:** Realiza las convoluciones y normalizaciones de lote.
- **Conexión de Salto:** Si el `stride` es 1, usamos `nn.Identity()`, que simplemente devuelve la entrada sin cambios. Si el `stride` es mayor a 1, usamos una convolución de $1 \times 1$ para que las dimensiones espaciales coincidan antes de la suma.
- **Fusión:** En el método `forward()`, sumamos ambas ramas y aplicamos la activación ReLU al final.



### 2. Construyendo la Arquitectura ResNet-34

Con nuestra unidad residual lista, la ResNet-34 se convierte simplemente en una secuencia organizada de estos módulos.

```python
class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        # Capas iniciales (Stem)
        layers = [
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]
        
        # Construcción dinámica de los bloques residuales
        prev_filters = 64
        for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
            stride = 1 if filters == prev_filters else 2
            layers.append(ResidualUnit(prev_filters, filters, stride=stride))
            prev_filters = filters
        
        # Capas finales de clasificación
        layers += [
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.LazyLinear(10), # 10 clases (ej. CIFAR-10)
        ]
        self.resnet = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.resnet(inputs)
```

Puntos clave del código:

- **Estructura de Bloques:** El bucle `for` gestiona la profundidad de la red: 3 unidades de 64 filtros, 4 de 128, 6 de 256 y 3 de 512.
- **Gestión del Stride:** Configuramos automáticamente un `stride=2` cada vez que el número de filtros aumenta, reduciendo así la resolución espacial de la imagen.
- **Eficacia:** Es sorprendente que con apenas 45 líneas de código podamos definir el modelo que ganó el desafío ImageNET 2015. Esto demuestra la elegancia de ResNet y la potencia expresiva de PyTorch.



Actividad

Ponte manos al código y aplica tus conocimientos realizando el notebook `EA2_S6_A2_ResNet34.ipynb` donde construiras y entrenaras una red ResNet34 desde cero.



Aunque es valioso saber construir estos modelos, en producción es preferible utilizar `torchvision.models`, donde estas arquitecturas ya están optimizadas y cuentan con pesos preentrenados en ImageNet. Puedes revisar los modelos disponibles en la documentación oficial https://docs.pytorch.org/vision/main/models.html



## Otras arquitecturas notables y tendencias actuales

Si bien **AlexNet**, **GoogLeNet** y **ResNet** establecieron los cimientos, la evolución de las arquitecturas no se detuvo allí. El enfoque de la investigación ha pasado de simplemente "hacer redes más profundas" a hacerlas más eficientes y fáciles de desplegar:

- **VGGNet (2014):** Aunque no fue la ganadora absoluta frente a GoogLeNet, popularizó el uso de filtros pequeños de $3 \times 3$ apilados, demostrando que la simplicidad y la profundidad pueden ser muy efectivas.
- **Xception / MobileNet:** Introdujeron las **convoluciones separables en profundidad** (*depthwise separable convolutions*), que reducen drásticamente el costo computacional, permitiendo que modelos de alta precisión corran en dispositivos móviles.
- **EfficientNet (2019):** Propuso un método de escalado compuesto que equilibra sistemáticamente la profundidad, el ancho y la resolución de la imagen, logrando una eficiencia máxima.
- **Vision Transformers (ViT):** La tendencia más reciente, que abandona las convoluciones por mecanismos de **atención**, tratando a las imágenes como secuencias de parches, similar a como se procesa el texto.