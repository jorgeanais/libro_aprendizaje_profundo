---
title: Red de primeros principios
subject: Aprendizaje Profundo
subtitle: Utilizando PyTorch
short_title: Principios de PyTorch
authors:
  - name: Jorge Anais
    affiliations:
      - Universidad de Antofagasta
      - Duoc UC
    orcid: 0000-0001-9051-1338
    email: jrganais@gmail.com
license: MIT
keywords: perceptron, funcion de activación
---

Anteriormente revisamos los conceptos teóricos que permiten que una red neuronal artificial aprenda a realizar una tarea. Hoy pondremos en acción este conocimiento para implenetar una red neuronal que sea capaz de distinguir entre imágenes de dígitos de `0` y `1`, extraídos del conjuto de de datos de [MNIST](https://en.wikipedia.org/wiki/MNIST_database). 


:::{figure} ../_images/numbero.png
:label: numbero
Ilustración de la representación de las imágenes de los dígitos y como cada pixel es guardado en un arreglo unidimensional.
:::

## Implementado una red neuronal desde primeros principios

Para ir soltando la mano, vamos a realizar un ejercicio que consiste en realizar una implementación de una red neuronal desde los primeros principios y utilizando PyTorch.

Nuestro objetivo es que la red aprenda una tarea que para nosotros puede resultad muy sencilla, sin embargo, para una máquina no tanto. Se trara del reconocimiento de dígitos escritos a mano.

## PyTorch

PyTorch es un framework de aprendizaje profundo de código abierto, desarrollado originalmente por el equipo de investigación de inteligencia artificial de Meta. Su principal atractivo radica en su diseño profundamente integrado con Python y en el uso de grafos computacionales dinámicos, lo que les permitirá construir, entrenar y depurar modelos de redes neuronales sobre la marcha de una manera muy intuitiva y flexible. Te invitamos a darle una ojeada a la [documentación](https://pytorch.org/projects/pytorch/).

### Funciones de activación

El primer paso será implementar las funciones de activación y sus respectivas derivadas. Aunque PyTorch ya incluye las funciones más comunes, programarlas desde cero nos ayudará a comprender su funcionamiento interno. Además, gracias al motor de diferenciación automática del *framework*, en la práctica no necesitaríamos calcular las derivadas manualmente. Sin embargo, lo haremos en esta ocasión con un fin puramente pedagógico, ya que es clave para asimilar a fondo el proceso de *backpropagation*.

**Definición de algunas funciones de activación y función de error, y la respectiva derivada**

| Función                  | Fórmula                                                      | Derivada                                                     |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Sigmoide                 | $\sigma(t) = \frac{1}{1 + e^{-t}}$                           | $\frac{d}{dt} \sigma(t)= \sigma(t)(1-\sigma(t))$             |
| Tangente hiperbólica     | $\tanh(t) = \frac{e^t - e^{-t}}{e^t + e^{-t}}$               | $ \frac{d}{dt} \tanh(t)= 1 - \tanh^2(t)$                     |
| Entropía cruzada binaria | $\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y_i}) - (1 - y_i) \log(1 - \hat{y_i}) \right] $ | $\frac{\partial}{\partial\hat{y}} \mathcal{L} = \frac{1}{N}(\hat{y} - y)$ |

Utilizando PyTorch definimos las funciones

```python
def sig(T: torch.Tensor):
    """Función de activación sigmoide"""
    return torch.reciprocal(1 + torch.exp(-1 * T))


def tanh(T: torch.Tensor):
    """Función de activación tangente hiperbólica."""
    E = torch.exp(T)
    e = torch.exp(-1 * T)
    return (E - e) * torch.reciprocal(E + e)


def bi_cross_ent_loss(y_pred, y, safe=True, epsilon=1e-7):
    """Función de pérdida de entropía cruzada binaria"""

    N = y.size()[0]  # tamaño del batch

    # Asegura que no haya valores indefinidos.
    if safe:
        y_pred = y_pred.clamp(epsilon, 1 - epsilon)

    B = (1-y) * torch.log(1 - y_pred) + y * torch.log(y_pred)
    return -1/N * torch.sum(B)
```



### Forward

Para construir la red, debemos definir sus parámetros, la propagación hacia adelante (*forward pass*) y la retropropagación del gradiente (*backpropagation*). El grafo de computación de la Figura # detalla la arquitectura que desarrollaremos: una red compuesta por dos capas ocultas. La primera utiliza una activación tangente hiperbólica (tanh) y la segunda emplea una función sigmoide. La capa de salida también utiliza una activación sigmoide, ya que nuestro objetivo es predecir las clases 0 y 1. Finalmente, la pérdida se calcula mediante la entropía cruzada.


:::{figure} ../_images/computation_graph_dense_nnv3.svg
:label: grafored
Grafo de computación de la red neuronal.
:::


Llevaremos esto a la práctica con el siguiente código. Para ello, creamos la clase `FFNN` que hereda de `torch.nn.Module` e inicializamos todos los parámetros (pesos y sesgos de las capas ocultas y de la capa de salida). Tengan en consideración que, por motivos pedagógicos, trabajaremos en un nivel de abstracción muy bajo:

```python
class FFNN(torch.nn.Module):
    def __init__(self, d0=784, d1=8, d2=4):
        """
        Crea la red FFNN con 2 capas ocultas y una capa de salida.
        d0: dimensión de la capa de entrada
        d1: número de neuronas de la primera capa oculta
        d2: número de neuronas de la segunda capa oculta
        """
        super(FFNN, self).__init__()

        # Crea los tensores como parámetros
        self.W1 = torch.nn.Parameter(torch.randn(d0, d1))
        self.b1 = torch.nn.Parameter(torch.zeros(d1))
        self.W2 = torch.nn.Parameter(torch.randn(d1,d2))
        self.b2 = torch.nn.Parameter(torch.zeros(d2))
        self.U  = torch.nn.Parameter(torch.randn(d2,1))
        self.c  = torch.nn.Parameter(torch.zeros(1))
    ...
```

Definimos dentro de la clase la propagación hacia adelante (forward pass)

```python
    ...
    def forward(self, x: torch.Tensor):
        # Calcula la pasada hacia adelante
        u1 = x @ self.W1 + self.b1
        h1 = tanh(u1)
        u2 = h1 @ self.W2 + self.b2
        h2 = sig(u2)
        u3 = h2 @ self.U + self.c
        y_pred = sig(u3)

        self._cache = [u1, u2]  # Guaradamos los valores de u1 y u2

        return y_pred
    ...
```

Ahora definiremos la retropropagación del gradiente. Este método recibe como argumentos los datos de entrada, las etiquetas reales y las predicciones de la red; con esta información, calcula los gradientes aplicando la regla de la cadena. No se abrumen por el cálculo manual de las derivadas: lo haremos por única vez con el propósito pedagógico de entender qué ocurre bajo el capó y por qué es un paso vital para que la red aprenda. En las próximas clases, dejaremos que PyTorch se encargue de todo este trabajo pesado por nosotros.

```python
    ...
    # Backpropagation
    def backward(self, x: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor):

        u1, u2 = self._cache  # recuperamos los valores de u1 y u2

        # tamaño del batch
        b = x.size()[0]

        # Estas son derivadas calculadas a mano
        dL_du3 = (1/b) * (y_pred - y)
        dL_dU  = sig(u2).t() @ dL_du3
        dL_dc  = torch.sum(dL_du3, 0)
        dL_dh2 = dL_du3 @ self.U.t()
        dL_du2 = dL_dh2 * sig(u2) * (1 - sig(u2))
        dL_dW2 = tanh(u1).t() @ dL_du2
        dL_db2 = torch.sum(dL_du2, 0)
        dL_dh1 = dL_du2 @ self.W2.t()
        dL_du1 = dL_dh1 * (1 - tanh(u1) * tanh(u1))
        dL_dW1 = x.t() @ dL_du1
        dL_db1 = torch.sum(dL_du1, 0)
    ...
```



### Preparación de los datos

Para ayudar a implementar el descenso de gradiente por lotes, PyTorch proporciona una clase llamada `DataLoader` en el módulo `torch.utils.data`. Esta puede cargar de manera eficiente lotes de datos del tamaño deseado.  `DataLoader` espera que el conjunto de datos esté representado como un objeto con al menos dos métodos: `__len__(self)` para obtener el número de muestras en el conjunto de datos, y `__getitem__(self, index)` para cargar la muestra en el índice dado (incluyendo el objetivo).

En el bloque de código a continuación, estamos creando nuestra propia clase `CustomDataSet`, la cual hereda de la clase base `Dataset` de PyTorch, para estructurar cómo nuestra red ingerirá los datos. 

En el método constructor (`__init__`), cargamos el archivo CSV, separamos las etiquetas de los píxeles y convertimos ambos a tensores. Aquí ocurre un paso crucial: normalizamos los píxeles dividiéndolos por 255 para que sus valores queden entre 0 y 1, lo cual facilita enormemente el entrenamiento. 

Finalmente, implementamos los dos métodos obligatorios que exige PyTorch: `__len__`, que informa el tamaño total del conjunto de datos, y `__getitem__`, que se encarga de extraer y entregar un ejemplo específico (la imagen aplanada y su etiqueta) cada vez que el `dataloader` lo solicite.



```python
class CustomDataSet(Dataset):
    def __init__(self, csv_path: Path):
        """Lee el archivo CSV con los datos y genera un Dataset"""
        df = pd.read_csv(csv_path)

        labels = torch.tensor(df["label"].values, dtype=torch.long)
        self.labels = labels.unsqueeze(1)  # shape: (N, 1)


        pixel_cols = [c for c in df.columns if c.startswith("pixel_")]
        pixels = df[pixel_cols].values.astype(np.float32) / 255.0  # Normalización
        self.flatten_images = torch.tensor(pixels)  # shape: (N, 784)

        self.num_features = len(pixel_cols)

    # Debemos definir __len__ para retornar el tamaño del dataset
    def __len__(self):
        return len(self.labels)

    # Debemos definir __getitem__ para retornar el i-ésimo ejemplo en nuestro dataset.
    def __getitem__(self, idx):
        flatten_image = self.flatten_images[idx]  # shape: (784,)
        label = self.labels[idx]

        return flatten_image, label
```



### Bucle de entrenamiento

Llegó el momento de utilizar nuestros datos para enseñar a la red neuronal. Para lograrlo, implementaremos un bucle iterativo que repetirá sistemáticamente los siguientes cinco pasos fundamentales:

1. Extraer un lote (*batch*): Tomar un subconjunto de nuestros datos de entrenamiento.
2. Propagación hacia adelante (*forward pass*): Pasar estos datos a través de la red para calcular sus predicciones.
3. Calcular la pérdida: Medir qué tan lejos están nuestras predicciones de las etiquetas reales utilizando la función de error.
4. Retropropagación (*backward pass*): Calcular los gradientes (las derivadas) del error con respecto a cada parámetro de la red.
5. Actualizar los pesos: Ajustar los parámetros utilizando el algoritmo de descenso del gradiente para que la red se equivoque menos en la siguiente iteración.


:::{figure} ../_images/ffnn_loop_simple.svg
:label: ffnn_loop_simple
Esquema del bucle de entrenamiento de la red neuronal artificial.
:::



A continuación la implementación en código.

```python
def loop_FFNN(
    dataset: Dataset,
    batch_size: int,   # tamaño del lote
    d1: int,  # número de neuronas en la capa 1
    d2: int,  # número de neuronas en la capa 2
    lr: float,  # tasa de aprendizaje
    epochs: int,
    run_in_GPU: bool=True,
    reports_every: int=1,
):
    # Define un tipo para los tensores según si correrá en la GPU o no
    device = 'cuda' if run_in_GPU else 'cpu'

    # d0 es la cantidad de `features` del dataset (pixeles de cada imagen)
    d0 = dataset.num_features

    # Cantidad de ejemplos
    N = len(dataset)

    # Instanciamos la red
    red = FFNN(d0, d1, d2)

    # Cargar la red en la GPU o CPU según elección
    red.to(device)

    # Mostrar la cantidad de parámetros
    print(f"Cantidad de parámetros: {red.num_parameters()}")

    # Crea un dataloader desde el dataset
    data = DataLoader(dataset, batch_size, shuffle=True)

    # Comienza el entrenamiento
    tiempo_epochs = 0
    for e in range(1, epochs + 1):
        inicio_epoch = time.process_time()

        for (x, y) in data:
            # Asegura de pasarlos a la GPU si fuera necesario
            x, y = x.to(device), y.to(device)

            # Computa la pasada hacia adelante (forward)
            y_pred = red.forward(x)

            # Computa la función de pérdida
            L = bi_cross_ent_loss(y_pred, y)

            # Computa los gradientes hacia atrás (backpropagation)
            red.backward(x, y, y_pred)


            # Descenso de gradiente para actualizar los parámetros
            for p in red.parameters():
                p.data = p.data - lr * p.grad

        tiempo_epochs += time.process_time() - inicio_epoch

        # Reporta el acierto cada "reports_every" cantidad de épocas
        if e % reports_every == 0:

            # Calcula la certeza de las predicciones sobre todo el conjunto
            X = dataset.flatten_images.to(device)
            Y = dataset.labels.to(device)

            # Predice usando la red
            Y_PRED = red.forward(X)

            # Calcula la pérdida de todo el conjunto
            L_total = bi_cross_ent_loss(Y_PRED, Y)

            # Elige una clase dependiendo del valor de Y_PRED
            Y_PRED_BIN = (Y_PRED >= 0.5).float()

            correctos = torch.sum(Y_PRED_BIN == Y).item()
            acc = (correctos / N) * 100

            sys.stdout.write(
                f"Epoch:{e:03d} Acc:{acc:.2f} Loss:{L_total:.4f} Tiempo/epoch:{tiempo_epochs/e:.3f}s\n"
            )


```

**Nota:** Para obtener la predicción final, aplicamos un umbral de 0.5 a la salida de la red. Es decir, si el valor arrojado por la función sigmoide es superior a 0.5, el dato se clasifica como la **clase 1**; de lo contrario, se le asigna la **clase 0**.



## Programando nuestra primera red neuronal

Realiza la actividad propuesta en el cuaderno interactivo `redPrimerosPrincipios.ipynb` y el respecto conjunto de datos. En este notebook se implementa en detalle lo que acabamos de revisar. Además plantea algunos experimentos que te permitirá profundizar en tu conocimiento acerca de las redes neuronales.

### Épocas y Lotes en Redes Neuronales

Para entrenar una red neuronal de forma eficiente, los datos se administran mediante dos conceptos fundamentales: las **épocas** (*epochs*) y los **lotes** (*batches*). Una época consiste en un ciclo completo donde la totalidad de los datos de entrenamiento pasa por la red neuronal una sola vez, abarcando los procesos de propagación hacia adelante, hacia atrás y la consecuente actualización de los parámetros. Sin embargo, procesar toda esta información de golpe no es computacionalmente óptimo. Por ello, durante cada época, el conjunto total de datos se divide en fracciones más pequeñas llamadas lotes. La red procesa un primer lote, actualiza sus pesos y luego continúa con el siguiente, repitiendo este ciclo hasta finalizar la época completa.

### El dilema del tamaño del lote

El tamaño que se elija para estos lotes tiene un impacto profundo tanto en la velocidad de entrenamiento como en el rendimiento final del modelo. Por un lado, utilizar lotes grandes tiene la ventaja de aprovechar al máximo la arquitectura de los aceleradores de hardware, como la memoria de video (VRAM) de las tarjetas gráficas (GPUs). Esto permite que el algoritmo procese una mayor cantidad de instancias por segundo, razón por la cual recomiendan llenar la memoria disponible. No obstante, esta práctica conlleva desventajas: los lotes muy grandes pueden generar inestabilidad, especialmente al inicio del entrenamiento o en modelos pequeños, y con frecuencia producen redes con una menor capacidad de generalización frente a datos nuevos.

## Subiendo de nivel de abstracción

Ahora realiza la actividad propuesta en el cuadernillo `red_estilo_pytorch.ipynb` y el respectivo conjunto de datos. Con el aprenderas a como definir la misma red utilizando un nivel de abstracción más alto.


## **Lectura complementaria** 

Para revisar conceptos con una mirada práctica se recomienda el libro *Hands-on machine learning with Scikit-Learn and PyTorch* por Aurélien Géron ( https://learning-oreilly-com.webezproxy.duoc.cl/library/view/hands-on-machine-learning/9798341607972/ ). El capítulo 10 introduce los conceptos asociados a como construir redes neuronales artificiales aplicado con PyTorch.

