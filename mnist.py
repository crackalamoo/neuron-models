import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_test[:8000]
y_train = y_test[:8000]
x_test = x_test[8000:]
y_test = y_test[8000:]
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(20, activation='relu', use_bias=False),
  tf.keras.layers.Dense(16, activation='relu', use_bias=False),
  tf.keras.layers.Dense(10, activation='relu', use_bias=False),
  tf.keras.layers.Dense(10, activation='softmax', use_bias=False)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='sgd',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=10, epochs=6)

model.evaluate(x_test,  y_test, verbose=2)
