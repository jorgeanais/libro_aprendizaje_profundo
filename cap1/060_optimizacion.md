---
title: Optimización de una red neuronal
subject: Aprendizaje Profundo
subtitle: Ajustando los parámetros de la red
short_title: Optimización
authors:
  - name: Jorge Anais
    orcid: 0000-0001-9051-1338
    email: jrganais@gmail.com
license: MIT
keywords: optimización, batch, adam
---

Para comenzar con el entrenamiento necesitamos asignar valores a los parámetros $\theta$ de la red . Nuestro primer acercamiento será asignar valores aleatorios. Es esperable que cuando realicemos por primera vez la propagación hacia adelante obtengamos una predicción totalmente incorrecta, y esto se reflejará en un error grande.

:::{important}
**Importante**


¿Cómo podemos lograr que el error disminuya? o dicho de otra manera, ¿cómo podemos modificar el valor de los parámetros para que el error sea el menor posible?
:::

La respuesta a esta interrogante constituye el núcleo del entrenamiento de las redes neuronales. A partir del cálculo del error (la diferencia entre el resultado esperado y el obtenido), ajustamos los parámetros de cada neurona de forma proporcional a su responsabilidad en la desviación final.

:::{tip}
**Analogía**


Para comprenderlo mejor, consideremos el siguiente ejemplo: imagina una empresa de relojes con una jerarquía clara: gerencia, ventas, y fabricación. Al detectar que los reportes de ventas son deficientes, el gerente general necesita identificar qué eslabón de la cadena está fallando para corregirlo. En este proceso, el flujo de optimización es inverso: la gerencia exige resultados a ventas, ventas reporta sus necesidades a fabricación, y fabricación ajusta sus procesos.
:::

## Descenso del gradiente

Ahora necesitamos "apretar las tuercas" de las neuronas que tienen mayor responsabilidad en el error, y así optimizar el modelo. Pero ¿cuánto tenemos que apretar las tuercas? o dicho de manera más formal ¿cuánto tengo que modificar los valores de los parámetros de la red para disminuir el error? La respuesta está en la técnica de optimización llamada "Descenso del gradiente". Ilustraremos este concepto con el siguiente ejemplo:

::::{hint}
**Ejemplo**


Por simplicidad, imaginemos una red neuronal con un único parámetro, el cual llamaremos $\theta$ (en general una red neuronales puede tener hasta miles de millones de parámetros). En la [](#loss_function_theta) podemos ver el valor de la pérdida de la red (error) en función del valor de este parámetro. Supongamos que la red está inicializada con el valor  $\theta = 15$ (equis roja en la parte derecha del gráfico). El objetivo del algoritmo de descenso del gradiente es encontrar un valor de $\theta$ tal que el valor de la pérdida sea lo más bajo posible. El caso ideal sería que el parámetro $\theta$ tenga un valor de 3, ya que es donde se alcanza el mínimo de la función.


:::{figure} ../_images/loss_function_theta_v4jorge.svg
:label: loss_function_theta
 Valor de la pérdida $\mathcal{L}$ en función del parámetro $\theta$. Se observa que el valor ideal (mínimo global) está en $\theta=3$. Se visualizan otros dos mínimos locales en $\theta=8$ y $\theta=13$. La X roja representa el valor inicial del parámetro.
:::
::::



El algoritmo de descenso del gradiente lo que hará será actualizar iterativamente el valor del parámetro $\theta$ para llevarlo a un valor mínimo que minimize la pérdida de la red ¿Cómo lo hace? actualizando el valor $\theta$ iterativamente. Primero se determina la dirección hacia donde se la pérdida disminuye (para ello se ocupa el gradiente) y luego se avanza en esa dirección una zancada ¿que tan grande es esa zancada? depende de un parámetro que se llama tasa de aprendizaje o *learning rate* en inglés.

Matemáticamente se formula de la siguiente manera:
$$
\theta_{\text{nuevo}} = \theta_{\text{viejo}} - \alpha \, \, \nabla\mathcal{L}(\theta)
$$
donde $\theta_{\text{nuevo}}$ es el valor actualizado en cada paso, $\theta_{\text{viejo}} $ es el valor antes de actualizar, $\alpha$ es la tasa de aprendizaje y $\nabla\mathcal{L}(\theta)$ es el gradiente. El signo negativo hace referencia a que nos movemos en la dirección contraria al gradiente, es decir, la dirección de máximo decrecimiento.

La [](#loss_gd_arcs), [](#loss_gd_arcs_large_lr) y [](#overshoot) ilustran el proceso de descenso del gradiente para diferentes valores de la tasa de aprendizaje.


:::{figure} ../_images/loss_gd_arcs.svg
:label: loss_gd_arcs
Aquí se ilustra como se actualiza el valor $\theta$ para un valor de **learning rate pequeño**. Se observa que se queda atrapado (converge) en el mínimo local $\theta_2$. Notamos que debido a que los pasos son pequeños requiere de varias iteraciones para converger.
:::


:::{figure} ../_images/loss_gd_arcs_large_lr.svg
:label: loss_gd_arcs_large_lr
Aquí se ilustra lo que pasa al aumentar el tamaño de la tasa d aprendizaje. En este caso el algoritmo se queda atrapado en en mínimo local $\theta_1$. Si bien no es ideal, tiene una pérdida muy comparable al mínimo global.
:::


:::{figure} ../_images/overshoot.png
:label: overshoot
Aquí se ilustra lo que pasa al aumentar el tamaño de la tasa de aprendizaje demasiado. En este caso el observamos que el valor salta de un lugar a otro sin lograr una convergencia. 
:::

El ejemplo que acabamos de revisar considera un único parámetro, pero en general, las redes neuronales pueden llegar a tener muchísimos parámetros ¡miles de millones! lo que hace sumamente complicado este tipo de optimización.


:::{warning}
**Importante**

El valor de la tasa de aprendizaje es un hiperparámetro crítico para encontrar una buena solución. Existen diferentes técnicas que permiten ajustar este parámetro dinámicamente.
:::

:::{hint}
**Comprobemos el aprendizaje**

Si consideramos una red neuronal densa con una capa de entrada de $784$ neuronas ($28 \times 28$ pixeles), dos capas ocultas de $128$ y $68$ neuronas respectivamente, y una capa de salida de $10$ neuronas ¿cuántos parámetros tenemos que optimizar?

a. Todos los pesos y sesgos de la red, es decir, se deben optimizar 109.942 parámetros.  
b. Solo los pesos de la última capa, es decir, se deben optimizar 680 parámetros.  
c. Solo un único valor, el parámetro $\theta$.  
:::


## Retropropagación

Ya tenemos un algoritmo que nos permite optimizar los parámetros de nuestra red, solo necesitamos definir una tasa de aprendizaje y conocer el gradiente. Recordemos que la tasa de aprendizaje es un hiperparámetro, sin embargo, el gradiente es algo un poquito más complicado de calcular. 

Pensemos que queremos actualizar el peso asociado a la primera primera capa oculta de una red densa. Para actualizar su valor necesitamos
$$
W^{(1)}_{\text{nuevo}} = W^{(1)} - \alpha \, \, \frac{\partial \mathcal{L}}{\partial W^{(1)}}
$$
Para actualizar cualquier parámetro de la red, por ejemplo, el peso asociado a la primera neurona de la primera capa oculta, necesitamos conocer el gradiente de la pérdida con respecto al peso de la primera neurona de la primera capa. Lo complejo es que este valor también depende de las capas ocultas posteriores y la capa de salida. Para ilustarar este punto, veamos el esquema de grafo de computación de la Figura #.


:::{figure} ../_images/computation_graph_dense_nnv2.1.svg
:label: computation_graph_dense_nnv2
Esquema del grafo de computación de una red neuronal con una capa oculta y una capa de salida. CE significa entropía cruzada.
:::


Necesitamos calcular el gradiente que le llega a $W^{(1)}$. Para calcular este valor se utiliza la **retropropagación del gradiente**. Esta técnica nos permite calcular el gradiente utilizando el grafo de computación. La Figura # contiene una ilustración como el error "fluye" desde la pérdida hacia el parámetro que se desea ajusta. Notar que para calcular el gradiente $\frac{\partial \mathcal{L}}{\partial W^{(1)}}$, necesito calcular todos los gradientes previos que se muestran en la imágen. Este procedimiento se ejecuta de manera totalmente automática mediante los motores de diferenciación implementados en las librerías de *Deep Learning*. A partir del grafo de computación y tras la propagación hacia adelante (*forward pass*), el software calcula los gradientes de forma inversa —desde la función de pérdida hacia la entrada—. Tras obtener estos valores, se procede a la optimización de los parámetros; la repetición iterativa de este ciclo constituye el **entrenamiento** de la red neuronal.


:::{figure} ../_images/backprop_gradiente_W1_v2.svg
:label: backprop_gradiente_W1_v2
 Representación de la retropropagación del gradiente hasta los pesos de la primera capa de la red neuronal densa.
:::


## Para profundizar...

A continuación se muestra como se calcula el gradiente $\frac{\partial \mathcal{L}}{\partial W^{(1)}}$ del ejemplo anterior, utilizando la regla de la cadena:
$$
\frac{\partial \mathcal{L}}{\partial W^{(1)}} = \frac{\partial \mathcal{L}}{\partial u^{(1)}}\frac{\partial u^{(1)}}{\partial W^{(1)}}
$$
pero a su vez,
$$
\frac{\partial \mathcal{L}}{\partial u^{(1)}} = \frac{\partial \mathcal{L}}{\partial h} \frac{\partial h}{\partial u^{(1)}}
$$
que a su vez se escribe
$$
\frac{\partial \mathcal{L}}{\partial h} = \frac{\partial \mathcal{L}}{\partial u^{(2)}} \frac{\partial u^{(2)}}{\partial h}
$$
lo que a su vez se puede escribir
$$
\frac{\partial \mathcal{L}}{\partial u^{(2)}} = \frac{\partial \mathcal{L}}{\partial \hat y} \frac{\partial \hat y}{\partial u^{(2)}}
$$
Estos gradientes que se han calculado aquí son los mismos que aparecen ilustrados en la Figura #. 

Sabías que Gottfried Leibniz formuló la regla de la cadena en 1676, aunque su aplicación específica en perceptrones multicapa no se documentaría hasta la década de 1960. 


## **Lectura complementaria** 

Para profundizar en el contenido de esta semana revisa el libro disponible gratuitamente *Alice’s Adventures in a differentiable wonderland* por Simone Scardapane (https://doi.org/10.48550/arXiv.2404.17625). En particular las secciones 2.2 y 2.3 permiten ahondar en los conceptos matemáticos vistos esta semana. 

Para complementar los contenidos revisados en esta sección de manera amena y didáctica, revisa el canal de Youtube "DotCSV". En particular el video ¿Qué es el Descenso del Gradiente? Algoritmo de Inteligencia Artificial | DotCSV https://youtu.be/A6FiCDoz8_4?si=OmMaD56LijIUjXtP.


## Revisemos lo aprendido

Analiza el caso presentado y elige la alternativa que mejor refleje lo aprendido esta semana: 

:::{hint}
**Pregunta**

Una red neuronal densa se entrena para clasificar imágenes de 28×28 píxeles en 10 categorías. La red tiene la siguiente arquitectura: capa de entrada → capa oculta 1 (128 neuronas, ReLU) → capa oculta 2 (64 neuronas, ReLU) → capa de salida (10 neuronas, Softmax).  
Después de la primera propagación hacia adelante con pesos inicializados aleatoriamente, se obtiene el siguiente vector de salida (probabilidades):  
```
[0.08, 0.07, 0.09, 0.41, 0.06, 0.05, 0.07, 0.06, 0.05, 0.06]
```
La etiqueta real de la imagen es el dígito **7**. Se utiliza descenso del gradiente con una tasa de aprendizaje *α* muy grande para actualizar los parámetros.  



**¿Cuál de las siguientes afirmaciones describe *correctamente* lo que ocurrirá durante el entrenamiento?**  
a. La retropropagación calculará el gradiente desde la capa de entrada hacia la capa de salida, y los parámetros se actualizarán una única vez para alcanzar el mínimo global de la pérdida.  
b. La función de pérdida calculará un error alto (la red asignó mayor probabilidad al dígito 3 que al 7), y una tasa de aprendizaje muy grande puede causar que el algoritmo no converja, saltando sobre el mínimo.  
c. Como la capa de salida usa Softmax, los gradientes no pueden retropropagarse hacia las capas ocultas, por lo que solo se actualizarán los pesos de la capa de salida.  
d. La tasa de aprendizaje es un parámetro de la red que se optimiza automáticamente durante la retropropagación, por lo que un valor inicial alto no representa ningún riesgo.  
e. Ninguna de las anteriores  
:::