from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf
import tensorflow_fold.public.blocks as td

# define the number of classes
NUM_LABELS = 10

# define input dimentions for mnist(28x28)
INPUT_LENGTH = 784

# define arguments
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_layers', 2, 'Number of hidden layers.')
flags.DEFINE_integer('num_units', 500, 'Number of units per hidden layer.')
flags.DEFINE_float('keep_prob', 0.75, 'Keep probability for dropout.')

# define td plan
td.define_plan_flags(default_plan_name='mnist')

# create plan function
def setup_plan(plan):
    # Convert the input Numpy array into a tensor.
    model_block = td.Vector(INPUT_LENGTH)

    # Create a placeholder for dropout, if we are in train mode
    keep_prob = (tf.placeholder_with_default(1.0, [], name='keep_prob') if plan.mode == plan.mode_keys.TRAIN else None)

    # add fc layers(using function) with drop out
    for _ in xrange(FLAGS.num_layers):
        model_block >>= td.FC(FLAGS.num_units, input_keep_prob=keep_prob)

    # add output layer with drop out
    model_block >>= td.FC(NUM_LABELS, activation=None, input_keep_prob=keep_prob)

    if plan.mode == plan.mode_keys.INFER:
        print("inference:")
        # compiler with model
        plan.compiler = td.Compiler.create(model_block)
        logits, = plan.compiler.output_tensors
    else:
        print("train:")
        # compiler with model, ground truth
        plan.compiler = td.Compiler.create(td.Record((model_block, td.Scalar(tf.int64))))
        logits, y_ = plan.compiler.output_tensors

    # prediction result
    y = tf.argmax(logits, 1)

    # load minist datasets
    datasets = tf.contrib.learn.datasets.mnist.load_mnist(FLAGS.logdir_base)

    if plan.mode == plan.mode_keys.INFER:
        plan.examples = datasets.test.images
        plan.outputs = [y]
    else:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
        plan.losses['cross_entropy'] = loss
        accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_), tf.float32))
        plan.metrics['accuracy'] = accuracy
        if plan.mode == plan.mode_keys.TRAIN:
            plan.examples = zip(datasets.train.images, datasets.train.labels)
            plan.dev_examples = zip(datasets.validation.images, datasets.validation.labels)

            # Turn dropout on for training, off for validation.
            plan.train_feeds[keep_prob] = FLAGS.keep_prob
        else:
            assert plan.mode == plan.mode_keys.EVAL
            plan.examples = zip(datasets.test.images, datasets.tesst.labels)


def main(_):
    assert 0 < FLAGS.keep_prob <= 1, '--keep_prob must be in (0, 1]'
    # create_from_flags:
    td.Plan.create_from_flags(setup_plan).run()

if __name__ == '__main__':
    tf.app.run()