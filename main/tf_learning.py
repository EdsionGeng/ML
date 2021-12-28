import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_eager_execution
dataset = tf.data.Dataset.from_tensor_slices(
    np.array([1.0, 2.0, 3.0, 4.0, 5.0])
)
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
sess = tf.Session()
for i in range(5):
    print(sess.run(one_element))
