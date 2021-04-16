# Лабораторная работа 1.

**Цель лабораторной работы:** Подготовить работоспособное окружение для решения
задачи классификации изображений из набора данных Food-101 с использованием
нейронных сетей глубокого обучения

**Задачи:**

1. С использованием примера[1] обучить предоставленную реализацию нейронной сети для решения задачи классификации изображений Food-101.
2. Создать и обучить сверточную нейронную сеть произвольной архитектуры с количеством сверточных слоев >3.

*Архитектура сети:*

   * Входные данные `inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))` 
   * Сверточный слой `x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(inputs)` 
   * Пулинг с методом выбора максимального значения `x = tf.keras.layers.MaxPool2D()(x)`
   * перевод многомерного тензера в одномерный `x = tf.keras.layers.Flatten()(x)`
   * Полносвязный слой, в задачу которого входит классификация. Параметрами заданы количество классов и активационная функция. `outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)`

```
 inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(inputs)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.Flatten()(x)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```

**Графики обучения нейронной сети с 1-м сверточным слоем**

***График метрики точности:***
![image](https://user-images.githubusercontent.com/56519328/115056336-5ece1780-9eeb-11eb-8170-18c6cce3bde3.png)
***График функции потерь:***
![image](https://user-images.githubusercontent.com/56519328/115056361-68f01600-9eeb-11eb-894e-cff2c88ff049.png)

**Графики обучения нейронной сети с 3-м сверточными слоями**

```
def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(inputs)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(x)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(x)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(x)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.Flatten()(x)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
  ```
  
***График метрики точности:***
![image](https://user-images.githubusercontent.com/56519328/115056878-03e8f000-9eec-11eb-8d02-538f3b413451.png)
***График функции потерь:***
![image](https://user-images.githubusercontent.com/56519328/115056917-119e7580-9eec-11eb-8984-c3a85c1d9468.png)

***Анализ полученных результатов***
В ходе работы модифицировали изначальную сеть добовлением 3-х сверточных слоев, что привело к увеличению глубины и времени обучения нейронной сети.
Проанализировав графики метрики точности, можно заметить, что увеличение глубины сети с 1-м сверточным слоем к улучшению точности не привело, а напротив - уменшило. 
Можно сделать общий вывод, что предложеная сеть не обучается на предложенных данных, это происходит по той причине, что для того чтобы хорошо обучить нейронную сеть с случайного начального приближения нам необходим очень большой датасет, а это миллионы данных(в нашем случае картинок) а то и десятки миллионов.
