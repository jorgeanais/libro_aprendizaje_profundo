---
title: Prediciendo Eventos Temporales con LSTM y GRU
subject: Aprendizaje Profundo
subtitle: Arquitecturas con memoria
short_title: LSTM y GRU
authors:
  - name: Jorge Anais
    orcid: 0000-0001-9051-1338
    email: jrganais@gmail.com
license: MIT
keywords: LSTM, GRU
---

Anteriormente revisamos las Redes Neuronales Recurrentes (RNN) y comprendimos que son el motor detrás del procesamiento de secuencias, como el lenguaje o el tiempo. Pero, ¿qué pasa cuando la "memoria" de nuestra red es frágil?

Imaginemos que están traduciendo un párrafo extenso. Para cuando llegan al punto final, han olvidado por completo cómo empezó la primera oración. Esto es exactamente lo que le ocurre a una RNN básica: a medida que los datos atraviesan la red, la información se transforma y se diluye.

El problema que nos enfrentamos es que en cada paso de tiempo, un poco de información se pierde. Después de unos pocos pasos, el estado de la red no conserva prácticamente ningún rastro de las entradas iniciales. En tareas complejas de largo alcance, la red se vuelve incapaz de conectar el principio con el final.

Para superar este "bloqueo", la arquitectura de las redes neuronales tuvo que evolucionar. No podíamos conformarnos con modelos que olvidaran el contexto de inmediato. Así nacieron las celdas especializadas en memoria a largo plazo.

Estas estructuras han sido tan revolucionarias que hoy en día las celdas básicas han quedado en el olvido (irónicamente). En este curso, exploraremos la arquitectura reina de esta categoría: La celda LSTM (Long Short-Term Memory) y las GRU. Estas redes que no solo procesan datos, sino que deciden qué recordar y qué olvidar, permitiendo que la inteligencia artificial entienda contextos profundos.

Para superar el olvido de las redes recurrentes tradicionales, entramos en el terreno de las arquitecturas con **compuertas de memoria**. Aquí es donde la magia ocurre, permitiendo que la red decida activamente qué debe ser recordado y que no.



## Celdas LSTM (Long Short-Term Memory)

Las LSTM no son solo una variante compleja de las RNN; son una solución de ingeniería al problema del **desvanecimiento del gradiente** (*vanishing gradient*). Mientras que una RNN estándar lucha por conectar información que ocurrió hace 10 pasos, una LSTM puede aprender dependencias de cientos de pasos de distancia.

El diagrama de la [](#lstm) muestra una representación de una LSTM de forma desplegada para los instantes justo anterior y el actual. A continuación se desarrolla una explicación de como funciona a nivel conceptual.

:::{figure} ../_images/cap3/lstm2.svg
:label: lstm
Arquitectura Interna y Flujo de Operaciones de una Celda LSTM. Este diagrama ilustra cómo una celda LSTM gestiona la información en un paso de tiempo $t$. La línea superior representa la "autopista" de la celda de memoria $c$, donde la información del pasado fluye casi sin alteraciones. La red procesa la entrada actual $x^{(t)}$ para generar un estado candidato $\tilde{c}^{(t)}$. Las compuertas $\gamma$, controladas por los parámetros entrenables $U$ y $V$, actúan como válvulas matemáticas para decidir dinámicamente qué fracción de información retener u olvidar en la memoria interna y qué información emitir como el estado oculto de salida $h^{(t)}$. Modificado de Jorge Pérez (2020).
:::







A diferencia de una RNN simple, una celda LSTM tiene una estructura interna más rica. Su funcionamiento se basa en dos componentes principales: la **celda de memoria** y las **compuertas**.

**La Celda de Memoria ($c^{(t)}$)**

Piensen en el estado de la celda ($c^{(t)}$) como una cinta transportadora o una autopista que recorre toda la cadena de la red de principio a fin. La información puede fluir a través de ella de manera lineal, permitiendo que los recuerdos del pasado remoto lleguen casi intactos al presente. La red solo interactúa con esta autopista a través de pequeñas interacciones controladas por las compuertas.

**Las Compuertas**

Las compuertas (representadas por $\gamma$) son operaciones matemáticas que actúan como "válvulas". Utilizan funciones de activación (como la sigmoide) que arrojan valores entre $0$ y $1$. Un $0$ significa "bloquea toda la información", y un $1$ significa "deja pasar todo".

Una celda LSTM tiene tres compuertas clave:

- **La Compuerta de Olvido ($\gamma_f^{(t)}$): \*¿Qué desechamos del pasado?\***

  Antes de procesar nueva información, la red debe hacer espacio. Esta compuerta analiza el estado oculto anterior ($h^{(t-1)}$) y el dato actual ($x^{(t)}$), y decide qué partes de la memoria antigua ($c^{(t-1)}$) ya no son útiles.

  - *Ejemplo práctico:* Si la red está analizando un texto y el sujeto cambia de "Juan" a "María", la compuerta de olvido se encargará de "borrar" el pronombre masculino de la memoria para prepararse para el femenino.

- **La Compuerta de Entrada ($\gamma_v^{(t)}$) y el Candidato ($\tilde{c}^{(t)}$): \*¿Qué nueva información guardamos?\***

  Aquí ocurren dos acciones en paralelo:

  1. Se crea un vector con "nueva información candidata" ($\tilde{c}^{(t)}$) a partir de los datos actuales.
  2. La **compuerta de entrada** decide qué fracción exacta de ese candidato es lo suficientemente importante como para ser añadida a nuestra autopista de memoria.

  La **actualización final de la memoria** se logra combinando lo que decidimos no olvidar y lo que decidimos agregar de nuevo:

  $$c^{(t)} = (\gamma_f^{(t)} \times c^{(t-1)}) + (\gamma_v^{(t)} \times \tilde{c}^{(t)})$$

- **La Compuerta de Salida ($\gamma_o^{(t)}$)**

  Nuestra celda de memoria ($c^{(t)}$) ahora contiene mucha información histórica y reciente, pero no toda es útil para el instante actual. La compuerta de salida filtra esta memoria masiva para extraer solo lo necesario y generar el **estado oculto** ($h^{(t)}$).

  Este estado oculto es la "cara pública" de la LSTM: es lo que la red utiliza para hacer su predicción en ese instante de tiempo específico y lo que le pasa como contexto a la siguiente celda en el futuro.

### ¿Por qué funcionan mejor que las RNN básicas?

- **Flujo del gradiente:** Al ser una operación principalmente aditiva (usando el símbolo $+$), el gradiente puede viajar hacia atrás en el tiempo sin ser multiplicado repetidamente por matrices de pesos que lo reduzcan a cero.
- **Escala de tiempo dinámica:** Gracias a que las compuertas son controladas por neuronas sigmoides, la red puede aprender por sí misma a mantener una unidad de memoria durante 5, 50 o 500 pasos de tiempo, cambiando su "constante de tiempo" según el contexto de la secuencia.


:::{warning}
**Dato clave** 

Aunque las LSTM tienen muchos más parámetros que una RNN simple (lo que las hace más lentas de entrenar por época), su capacidad para aprender dependencias complejas compensa con creces el costo computacional, lo que la convertía en la arquitectura estándar para audio, traducción y series temporales hasta antes de la llegada de los Transformers.
:::


## Celdas GRU (Gated Recurrent Unit)

La GRU (2014) es una versión simplificada y altamente eficiente de la LSTM. Es como una "LSTM optimizada" que suele ofrecer un rendimiento similar pero con menos potencia de cómputo.



:::{figure} ../_images/cap3/GRU2.svg
:label: gru
Arquitectura simplificada y flujo de información en una Unidad Recurrente con Compuertas (GRU).El diagrama detalla el flujo del estado oculto $h^{(t)}$ en una celda GRU. A diferencia de la LSTM, la GRU carece de una celda de memoria independiente y consolida la información en un único canal. La compuerta de reinicio $\rho^{(t)}$, indicada por la doble barra transversal, modula la influencia del estado anterior al calcular la nueva propuesta o estado candidato $\tilde{h}^{(t)}$. Posteriormente, la compuerta de actualización $\gamma^{(t)}$ aplica una interpolación lineal (representada por las líneas curvas paralelas) para establecer el balance exacto entre retener el estado histórico $h^{(t-1)}$ y adoptar la nueva información, determinando así la salida de la celda. Modificado de Jorge Perez (2020)
:::




Si observamos el diagrama de la [](#gru), la diferencia más notable con la LSTM es que la GRU elimina la celda de memoria separada $c^{(t)}$. En su lugar, toda la información, tanto la memoria a largo plazo como la predicción a corto plazo, viaja a través de un único canal: el estado oculto $h^{(t)}$. Modificado de Jorge Pérez (2020).

Para gestionar este flujo sin que la información colapse, la GRU condensa las operaciones en solo dos compuertas principales:

### La Compuerta de Reinicio $\rho^{(t)}$

Esta compuerta (representada por el símbolo de interrupción transversal cerca de $\rho^{(t)}$ se sitúa entre el estado anterior $h^{(t-1)}$ y el cálculo del nuevo estado candidato.

Su función es decidir cuánta información del paso anterior es relevante para la entrada actual ($x^{(t)}$).

- Si $\rho^{(t)}$ es cercano a $0$, la red "corta" la conexión con el pasado y evalúa la entrada actual de forma aislada.
- Si $\rho^{(t)}$ es cercano a $1$, la red trae todo el contexto histórico para interpretar el presente.
- En la práctica, si estamos traduciendo un texto técnico complejo y cambiamos abruptamente de un párrafo sobre astrofísica a uno sobre metodologías de evaluación, esta compuerta ayuda a "reiniciar" el contexto semántico de la red para no mezclar conceptos.

### El Estado Candidato $\tilde{h}^{(t)}$

Con el filtro del pasado ya aplicado, la red genera una nueva propuesta de memoria combinando la entrada actual, $x^{(t)}$ ponderada por la matriz $U$,  con la fracción del pasado que decidimos conservar.

### La Compuerta de Actualización $\gamma^{(t)}$

Aquí reside la gran elegancia de la GRU. En el diagrama, las líneas curvas dobles en la parte superior representan una **interpolación lineal** (o transición cruzada) controlada por $\gamma^{(t)}$.

Esta compuerta fusiona las antiguas compuertas de *Olvido* y *Entrada* de la LSTM en un solo control deslizante. Decide qué porcentaje del estado final será información antigua y qué porcentaje será información nueva:
$$
h^{(t)} = (\gamma^{(t)} \times \tilde{h}^{(t)}) + (1 - \gamma^{(t)}) \times h^{(t-1)}
$$


- Si $\gamma^{(t)} \approx 1$, el nuevo estado candidato reemplaza casi por completo a la memoria anterior.
- Si $\gamma^{(t)} \approx 0$, la red ignora lo que está leyendo ahora y mantiene intacta la memoria del pasado.

Una celda GRU hace un trabajo excepcional al equilibrar la retención de datos históricos y la adaptación a nuevos inputs, utilizando menos recursos computacionales. Esta eficiencia la convierte en una opción predilecta en la industria moderna de *Data Science* para procesar secuencias de texto, señales de audio o series de tiempo donde el tiempo de entrenamiento es un recurso crítico.

## Tip sobre rendimiento: ¿CPU o GPU?

El entrenamiento de redes recurrentes presenta desafíos particulares de rendimiento que debes considerar:

- **Modelos pequeños:** Para las RNN simples que tienen pocos parámetros, el entrenamiento suele ser más rápido en una CPU multinúcleo. Esto se debe a que las operaciones son multiplicaciones de matrices pequeñas que ocurren dentro de un ciclo `for` (el paso del tiempo), lo cual es difícil de paralelizar en una GPU.
- **Modelos grandes (LSTM/GRU):** Si decides escalar a modelos más complejos, la GPU es fundamental. En entornos como Google Colab (Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU T4). Por ejemplo, Keras utiliza núcleos cuDNN de NVIDIA, que son implementaciones de bajo nivel altamente optimizadas.


:::{tip}
**Actividad:**

Entender conceptualmente cómo las celdas LSTM y GRU deciden qué información retener y cuál desechar es imporante, pero el verdadero aprendizaje se logra llevando estos conceptos al código. En el cuadernillo  `EA3_S8_A4_LSTM_GRU_PyTorch.ipynb` pondremos a prueba estas arquitecturas entrenando modelos para resolver problemas con secuencias reales.
:::


**Referencias** 

- François Chollet (2021). Deep Learning with Python. MANNING [Libro]. https://www.manning.com/books/deep-learning-with-python-second-edition
- Géron, A. (2025). Hands-on machine learning with Scikit-Learn and PyTorch. O'Reilly Media. [Libro] https://www.oreilly.com/library/view/hands-on-machine-learning/9798341607972/ 

- Goodfellow, I., Bengio, Y., y Courville, A. (2016). Deep learning. MIT Press. [Libro] [http://www.deeplearningbook.org](http://www.deeplearningbook.org/). 

- Scardapane, S. (2024). Alice’s adventures in a differentiable wonderland. [Libro] https://doi.org/10.48550/arXiv.2404.17625. 

**Lecturas** 

- Construyendo y entrenando redes neuronales MLP. Géron, A. (2025). Hands-on machine learning with Scikit-Learn and PyTorch. Capítulo 13: Processing Sequences Using RNNs and CNNs. 
- François Chollet (2021). Deep Learning with Python. Capítulo 10: Deep learning for timeseries.
- Goodfellow (2016). Deep learning. MIT Press. Capítulo 10 Sequence Modeling: Recurrent and Recursive Nets