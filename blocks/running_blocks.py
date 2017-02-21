from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
sess = tf.InteractiveSession()
import tensorflow_fold.public.blocks as td

# evaluating indivisual inputs
print("one hot vector ======")
onehot_block = td.OneHot(5)

# evaluating block using eval
print(onehot_block.eval(3))
# => array([0., 0., 0., 1., 0.], dtype=float32)

# others
print(onehot_block.eval(0))
print(onehot_block.eval(1))
print(onehot_block.eval(2))
print(onehot_block.eval(4))


print("composite blocks =====")
composite_blocks = td.Scalar() >> td.AllOf(td.Function(tf.negative), td.Function(tf.square))
print(composite_blocks.eval(2)[0]) #negative
print(composite_blocks.eval(2)[1]) #square

print("batching inputs =====")
# Compiler compiles out model down to TensorFlow graph.
# Compiler will also do type inference and validation on the model.
# The outputs are ordinary TensorFlow tensors, which can be connected to Tensorflow loss function and optimizer.

print("blocks have associated input and output types =====")
print(td.Scalar().input_type)
scalar_block = td.Scalar()
print(td.Scalar().output_type)
print(scalar_block.eval(5))

print("type inference =====")



