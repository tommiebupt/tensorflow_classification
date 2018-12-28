import tensorflow as tf
from tensorflow.python.framework import ops,dtypes
from tensorflow.python.ops import array_ops,variables


def _createLocalVariable(name, shape, collections=None, validate_shape=True,
              dtype=dtypes.float32):
  """Creates a new local variable.
  """
  # Make sure local variables are added to 
  # tf.GraphKeys.LOCAL_VARIABLES
  collections = list(collections or [])
  collections += [ops.GraphKeys.LOCAL_VARIABLES]
  return variables.Variable( 
		initial_value=array_ops.zeros(shape, dtype=dtype),
  		name=name,
  		trainable=False,
  		collections=collections,
  		validate_shape=validate_shape)

def streamingConfusionMatrix(label, prediction, weights=None, num_classes=None):
  """
  Compute a streaming confusion matrix
  :param label: True labels
  :param prediction: Predicted labels
  :param weights: (Optional) weights (unused)
  :param num_classes: Number of labels for the confusion matrix
  :return: (percentConfusionMatrix,updateOp)
  """
  # Compute a per-batch confusion

  batch_confusion = tf.confusion_matrix(label, prediction,
                                    num_classes=num_classes,
                                    name='batch_confusion')

  count = _createLocalVariable(None,(),dtype=tf.int32)
  confusion = _createLocalVariable('streamConfusion',[num_classes, num_classes],dtype=tf.int32)

  # Create the update op for doing a "+=" accumulation on the batch
  countUpdate = count.assign(count + tf.reduce_sum(batch_confusion))
  confusionUpdate = confusion.assign(confusion + batch_confusion)

  updateOp = tf.group(confusionUpdate,countUpdate)

  #percentConfusion = 100 * tf.truediv(confusion,count)
  #return percentConfusion,updateOp
  return confusion,updateOp
