import tensorflow as tf

print(tf.__version__)
print(tf.config.experimental.get_visible_devices('GPU'))
a = tf.constant(1)
b = tf.constant(2)
print(a+b)
