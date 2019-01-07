from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import tensorflow as tf
import numpy as np
from nets import nets_factory
slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

class OptimizeGraph(object):
  def __init__(self, 
    inputs, 
    labels, 
    num_classes, 
    num_samples_per_epoch,#for learning_rate_decay.
    model_name = "resnet_v1_50", 
    labels_offset = 0,
    trainable_scopes = None,
    label_smoothing = 0.0,
    optimizer_type = 'adam',
    moving_average_decay = None, #smooth variable.
    weight_decay = 0.00004,      #regularization dept.
    learning_rate_decay_type = 'exponential'):

    self._input = inputs #tf.placeholder(tf.float32, [None, None, None])
    self._labels = labels #tf.placeholder(tf.float32, [None, None])
    self._model_name = model_name
    self._num_classes = num_classes
    self._labels_offset = labels_offset
    self._trainable_scopes = trainable_scopes 

    #DEFAULT
    self._label_smoothing = label_smoothing
    self._optimizer_type = optimizer_type
    self._moving_average_decay = moving_average_decay
    self._weight_decay = weight_decay
    self._num_samples_per_epoch = num_samples_per_epoch
    self._learning_rate_decay_type = learning_rate_decay_type

    self._all_update_ops = []
    self._all_summaries = set()

    #RETURN
    self._train_op = None
    self._summary_op = None
    self._build_network()
  #enddef
  
  def train_ops(self):
    return self._train_op, self._summary_op
  #enddef

  def _build_network(self):
    #with tf.Graph().as_default():
      #Gather initial summaries.
      self._all_summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))

      #################
      #  stage 1
      #################
      #Build network.
      print("model_name:", self._model_name)
      self._network_fn = nets_factory.get_network_fn(self._model_name,
          self._num_classes-self._labels_offset,
          self._weight_decay,
          is_training = True)
      #Forward network.
      self._logits, self._end_points = self._network_fn(self._input)
      self._probabilities = tf.nn.softmax(self._logits, axis=-1, name="probabilities")
      self._predictions = tf.argmax(self._logits, axis=-1, name="predictions")
      self._build_and_gather_losses()

      #Gather update_ops, eg.the updates for the batch_norm variable.
      self._all_update_ops.append(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
      
      #Add summaries.
      for variable in slim.get_model_variables():
        self._all_summaries.add(tf.summary.histogram(variable.op.name, variable))
      for end_point in self._end_points:
        x = self._end_points[end_point]
        self._all_summaries.add(tf.summary.histogram('activations/'+end_point,x))
        #self._all_summaries.add(tf.summary.scalar('sparsity/'+end_point, 
        #    tf.nn.zero_fraction(x)))

      #################
      # stage 2
      #################
      self._configure_variable_moving_average() #moving averge for each model variable.
      self._global_step = tf.train.create_global_step()
      self._configure_learning_rate() #require self._global_step.
      self._configure_optimization() #require self._learning_rate.
      self._get_variables_to_train()

      #require self._variables_to_train,self._total_loss,self._optimizer,self._global_step.
      self._optimize_variables()
      
      self._merge_all_update_ops()
      self._merge_all_summaries()      
  #enddef
  
  def _configure_variable_moving_average(self):
    if self._moving_average_decay:
      self._moving_average_variables = slim.get_model_variables()
      self._variable_averages = tf.train.ExponentialMovingAverage(self._moving_average_decay, 
            self._global_step)
      self._all_update_ops.append(self._variable_averages.apply(self._moving_average_variables)) 
    else:
      self._moving_average_variables, self._variable_averages = None, None
  #enddef

  def _configure_learning_rate(self):
    decay_steps = int(self._num_samples_per_epoch*FLAGS.num_epochs_per_decay/FLAGS.batch_size)
    if self._learning_rate_decay_type == 'exponential':
      self._learning_rate =  tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                      self._global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
    elif self._learning_rate_decay_type == 'fixed':
      self._learning_rate = tf.constant(FLAGS.initial_learning_rate, name='fixed_learning_rate')
    elif self._learning_rate_decay_type == 'polynomial':
      self._learning_rate = tf.train.polynomial_decay(FLAGS.initial_learning_rate,
                                     self._global_step,
                                     decay_steps,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
    else:
      raise ValueError('learning_rate_decay_type [%s] was not recognized' %
                        self._learning_rate_decay_type)    
    #endif

    self._all_summaries.add(tf.summary.scalar('learning_rate', self._learning_rate))
  #enddef

  
  def _configure_optimization(self):
    if self._optimizer_type == 'adadelta':
      self._optimizer = tf.train.AdadeltaOptimizer(
              self._learning_rate,
              rho=0.95,
              epsilon=1.0)
    elif self._optimizer_type == 'adagrad':
      self._optimizer = tf.train.AdagradOptimizer(
              self._learning_rate,
              initial_accumulator_value=0.1)
    elif self._optimizer_type == 'adam':
      self._optimizer = tf.train.AdamOptimizer(
              self._learning_rate,
              beta1=0.9,
              beta2=0.999,
              epsilon=1.0)
    elif self._optimizer_type == 'ftrl':
      self._optimizer = tf.train.FtrlOptimizer(
              self._learning_rate,
              learning_rate_power=-0.5,
              initial_accumulator_value=0.1,
              l1_regularization_strength=0.0,
              l2_regularization_strength=0.0)
    elif self._optimizer_type == 'momentum':
      self._optimizer = tf.train.MomentumOptimizer(
              self._learning_rate,
              momentum=0.9,
              name='Momentum')
    elif self._optimizer_type == 'rmsprop':
      self._optimizer = tf.train.RMSPropOptimizer(
              self._learning_rate,
              decay=0.9,
              momentum=0.9,
              epsilon=1.0)
    elif self._optimizer_type == 'sgd':
      self._optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
      raise ValueError('Optimizer [%s] was not recognized' % self._optimizer_type)
  #enddef

  def _get_variables_to_train(self):
    if self._trainable_scopes is None:
      self._variables_to_train =  tf.trainable_variables()
      return
    else:
      scopes = [scope.strip() for scope in self._trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
      variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
      variables_to_train.extend(variables)
    self._variables_to_train =  variables_to_train
  #enddef
  
  def _build_and_gather_losses(self):
    #Build losses.
    #labels = slim.one_hot_encoding(self._labels-self._labels_offset, 
    #     self._num_classes-self._labels_offset)
    labels = slim.one_hot_encoding(self._labels, self._num_classes)
    if 'AuxLogits' in self._end_points:
      slim.losses.softmax_cross_entropy(
                  self._end_points['AuxLogits'], labels,
                  label_smoothing=self._label_smoothing, weights=0.4,
                  scope='aux_loss')
    slim.losses.softmax_cross_entropy(
      self._logits, labels, label_smoothing=self._label_smoothing, weights=1.0)

    ##Focal Loss
    """
    weights = tf.gather_nd(tf.nn.softmax(self._logits),
      tf.concat([tf.expand_dims(tf.range(labels.get_shape().as_list()[0]),-1), 
      tf.expand_dims(tf.cast(tf.argmax(labels, -1), tf.int32),-1)], 1))  
    pow_weights = tf.square(1- weights)
    tf.losses.softmax_cross_entropy(
      labels, self._logits, label_smoothing=self._label_smoothing, weights=pow_weights)
    """
    #Gather losses.  
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regularization_loss = tf.add_n(regularization_losses, name='regularization_loss')
    losses = tf.get_collection(tf.GraphKeys.LOSSES)
    loss = tf.add_n(losses, name='loss')
    self._total_loss = tf.add_n([loss,regularization_loss])

    #Add summaries.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES):
      self._all_summaries.add(tf.summary.scalar('losses/%s'%loss.op.name, loss))
    if loss is not None:
      self._all_summaries.add(tf.summary.scalar('/'.join(filter(None,
        ['Losses', 'loss'])), loss))
    if regularization_loss is not None:
      self._all_summaries.add(tf.summary.scalar('Losses/regularization_loss', 
        regularization_loss))
  #enddef

  def _optimize_variables(self):
    #[(gradient, variable),...]
    grads_and_vars = self._optimizer.compute_gradients(self._total_loss, 
      var_list=self._variables_to_train)

    #clip gradient.
    grads, variables = zip(*grads_and_vars)
    grads, global_norm = tf.clip_by_global_norm(grads, 5)
    self._grads_and_vars = zip(grads, variables)

    grad_updates = self._optimizer.apply_gradients(self._grads_and_vars, 
          global_step=self._global_step)
    self._all_update_ops.append(grad_updates)
    
    for grad, var in self._grads_and_vars:
      if grad is not None:
        if isinstance(grad, tf.IndexedSlices):
          grad_values = grad.values
        else:
          grad_values = grad
        self._all_summaries.add(tf.summary.histogram(var.op.name+'_gradient', grad_values))
        self._all_summaries.add(tf.summary.histogram(var.op.name+'_gradient_norm',
            tf.global_norm([grad_values])))
      else:
        tf.logging.info('Var %s has no gradient', var.op.name)
  #enddef

  def _merge_all_update_ops(self):
    update_op = tf.group(*self._all_update_ops)
    with tf.control_dependencies([update_op]):
      self._train_op = tf.identity(self._total_loss, name='train_op')
  #enddef

  def _merge_all_summaries(self):
    self._all_summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    self._summary_op = tf.summary.merge(list(self._all_summaries), name='summary_op')
  #enddef
#endclass


if __name__ == "__main__":
  labels = tf.cast(tf.convert_to_tensor(np.random.randint(2, size=100)),tf.int32)
  images = tf.cast(tf.convert_to_tensor(np.random.random([100, 256, 256, 3])),tf.float32)
  
  train_op, summary_op = OptimizeGraph(images, labels, 2, 100).train_ops()

  with tf.Session() as sess:
    init_op = tf.group([tf.global_variables_initializer(),
        tf.local_variables_initializer(), tf.tables_initializer()])
    sess.run(init_op)
    total_loss, _ =sess.run([train_op,summary_op])
    print(total_loss)
