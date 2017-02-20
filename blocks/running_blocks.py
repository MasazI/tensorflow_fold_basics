from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
sess = tf.InteractiveSession()
import tensorflow_fold.public.blocks as td

# evaluating indivisual inputs
onehot_block = td.OneHot(5)

# evaluating block using eval
print(onehot_block.eval(3))
# => array([0., 0., 0., 1., 0.], dtype=float32)

# others
print(onehot_block.eval(0))
print(onehot_block.eval(1))
print(onehot_block.eval(2))
print(onehot_block.eval(4))

