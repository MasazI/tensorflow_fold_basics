from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf
sess = tf.InteractiveSession()
import tensorflow_fold.public.blocks as td

# Map and Reduce
print(td.Map(td.Scalar()))
print((td.Map(td.Scalar()) >> td.Reduce(td.Function(tf.multiply))).eval(range(1,10)))

# simple fc network
def reduce_net_block():
    net_block = td.Concat() >> td.FC(20) >> td.FC(1, activation=None) >> td.Function(lambda xs: tf.squeeze(xs, axis=1))
    return td.Map(td.Scalar()) >> td.Reduce(net_block)

# generate ramdom example data, result
def random_example(fn):
    length = random.randrange(1, 10)
    data = [random.uniform(0,1) for _ in range(length)]
    result = fn(data)
    return data, result

# indeterminant data length
print(random_example(sum))
print(random_example(sum))
print(random_example(sum))
print(random_example(sum))
# another function
print(random_example(min))
print(random_example(min))
print(random_example(min))
print(random_example(min))
print(random_example(min))

# train using simple fc network with indeterminant data
def train(fn, batch_size=100):
    net_block = reduce_net_block()
    compiler = td.Compiler.create((net_block, td.Scalar()))

    # compiler have 2 tds. output_tensor for each tds.
    y, y_ = compiler.output_tensors
    loss = tf.nn.l2_loss(y - y_)
    train = tf.train.AdamOptimizer().minimize(loss)
    sess.run(tf.global_variables_initializer())
    validation_fd = compiler.build_feed_dict(random_example(fn) for _ in range(1000))
    for i in range(2000):
        sess.run(train, compiler.build_feed_dict(random_example(fn) for _ in range(batch_size)))
        if i % 100 == 0:
            print(i, sess.run(loss, validation_fd))
    return net_block

# execute train
sum_block = train(sum)

# execute evaluation
print(sum_block.eval([1, 1]))
print(sum_block.eval([1, 8]))
print(sum_block.eval([1, 1000]))