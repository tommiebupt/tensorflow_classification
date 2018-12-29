from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import tensorflow as tf
#from train_graph import TrainGraph 
from input_graph import *
from image_processing import preprocess_image
from optimizer import OptimizeGraph

slim = tf.contrib.slim
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', os.path.abspath('./output/tfmodel'),
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_num_steps', 10000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('image_size', 256,
                            """Provide square images of this size.""")
tf.app.flags.DEFINE_boolean('fine_tune_imagenet', False,
                            """If set, randomly initialize the final layer """
                            """of weights in order to train the network on a """
                            """new task.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', './output/pretrain/resnet_v1_50.ckpt',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,#0.1,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_integer('batch_size', 32,
                          """Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
                          """Learning rate decay factor.""")


class Train(object):
  def __init__(self, dataset, 
    checkpoint_exclude_scopes = "resnet_v1_50/logits",
    ignore_missing_vars = False, 
    log_every_n_steps = 10, 
    save_summaries_secs = 100, 
    save_interval_secs = 1000, 
    per_process_gpu_memory_fraction=1.0):
    
    self._dataset = dataset
    self._checkpoint_exclude_scopes = checkpoint_exclude_scopes 
    self._ignore_missing_vars = ignore_missing_vars
    self._log_every_n_steps = log_every_n_steps 
    self._save_summaries_secs = save_summaries_secs 
    self._save_interval_secs = save_interval_secs 

    self._session_config = tf.ConfigProto()
    self._session_config.gpu_options.allow_growth = True
    self._session_config.gpu_options.per_process_gpu_memory_fraction = \
             per_process_gpu_memory_fraction 

    #
    self.get_train_ops()
    
  #enddef

  def get_train_ops(self):
    image_paths, labels = self._dataset.get_image_paths()

    train_data= PathsInputforTrain()  
    image_batch, label_batch = train_data.get_batches(image_paths,
        labels, 
        preprocess_image, 
        FLAGS.image_size,
        FLAGS.batch_size)
        
    self._train_op, self._summary_op =\
       OptimizeGraph(image_batch, 
        label_batch, 
        self._dataset.get_num_classes(),
        self._dataset.get_num_samples()).train_ops()

  def _init_model(self):
    if FLAGS.pretrained_model_checkpoint_path is None:
      return None

      # Warn the user if a checkpoint exists in the train_dir. Then we'll be
      # ignoring the checkpoint anyway.
      if tf.train.latest_checkpoint(FLAGS.train_dir):
        tf.logging.info( 
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
              % FLAGS.train_dir)
        return None
    
    #############
    # Fine Tune
    #############
      exclusions = []
      if self._checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in self._checkpoint_exclude_scopes.split(',')]

      # TODO(sguada) variables.filter_variables()
      variables_to_restore = []
      restore_flag = True
      variables_to_restore = slim.get_model_variables(exclude=exclusions)
      for var in slim.get_model_variables():
        for exclusion in exclusions:
          if var.op.name.startswith(exclusion):
            restore_flag = False
            break
        if restore_flag:
          variables_to_restore.append(var)
        restore_flag = True

      if tf.gfile.IsDirectory(FLAGS.pretrained_model_checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.pretrained_model_checkpoint_path)
      else:
        checkpoint_path = FLAGS.pretrained_model_checkpoint_path

      tf.logging.info('Fine-tuning from %s' % checkpoint_path)

      return slim.assign_from_checkpoint_fn(
            checkpoint_path,
            variables_to_restore,
            ignore_missing_vars=self._ignore_missing_vars)
  #enddef

  def _train_step(self, sess, train_op, global_step, train_step_kwargs):
    start_time = time.time()
    total_loss, np_global_step = sess.run([train_op, global_step])
    time_elapsed = time.time() - start_time

    if 'should_log' in train_step_kwargs:
      if sess.run(train_step_kwargs['should_log']):
        tf.logging.info('global step %d: loss = %.4f (%.3f sec/step)',
          np_global_step, total_loss, time_elapsed)
    
    if 'should_stop' in train_step_kwargs:
      should_stop = sess.run(train_step_kwargs['should_stop'])
    else:
      should_stop = False

    return total_loss, should_stop
  #enddef



  def start(self):
    if self._train_op is None:
      raise ValueError('train_op cannot be None.')
    
    max_num_steps = FLAGS.max_num_steps
    if max_num_steps is not None and max_num_steps <=0:
      raise ValueError('`max_num_of_steps` must be either None or a positive number.')

    graph = tf.get_default_graph()
    with graph.as_default():
      global_step = tf.train.get_or_create_global_step()
      saver = tf.train.Saver(max_to_keep=0)

      with tf.name_scope('init_ops'):
        init_op = tf.global_variables_initializer()
        
        #1-D tensor: names of uninitialized variables.
        #ready_op = tf.report_uninitialized_variables()

        local_init_op = tf.group(
          tf.local_variables_initializer(), #ops: initialize all local variables. 
          tf.tables_initializer()  #ops:initialize all tables. NoOp if nonexists.
        )

      with tf.name_scope('train_step'):
        train_step_kwargs = {}
        if max_num_steps:
          should_stop_op = tf.greater_equal(
            global_step, 
            max_num_steps)
        else:
          should_stop_op = tf.constant(False)

        train_step_kwargs['should_stop'] = should_stop_op


        if self._log_every_n_steps > 0:
          train_step_kwargs['should_log'] = tf.equal(
              tf.mod(global_step, 
              self._log_every_n_steps), 
              0)

        train_step_kwargs['logdir'] = FLAGS.train_dir 

      sv = tf.train.Supervisor(graph=graph, 
        is_chief = True, 
        logdir = FLAGS.train_dir,
        init_op = init_op, 
        #check whether model ready or not(gloabl/local vars initialized).
        #ready_op=ready_op, 
        #check whether model is ready or not(self-produced local vars init).
        #ready_for_local_init_op, 
        local_init_op = local_init_op,
        summary_op = self._summary_op, 
        global_step = global_step, 
        saver = saver,
        save_summaries_secs = self._save_summaries_secs,
        save_model_secs = self._save_interval_secs,
        init_fn = self._init_model())

      train_step_kwargs['summary_writer'] = sv.summary_writer
      total_loss = None
      should_try = True
      while should_try:
        try:
          should_try = False
          with sv.managed_session('', 
            start_standard_services=False, 
            config=self._session_config) as sess:

            print("----------------------------------------------------------")
            tf.logging.info('Start Session')
            if FLAGS.train_dir:
              #start summary/savemodel/stepcounter threads.
              sv.start_standard_services(sess)
            threads = sv.start_queue_runners(sess)
            tf.logging.info('Start Queues.')

            try:
              while not sv.should_stop():
                total_loss, should_stop = self._train_step(
                  sess, 
                  self._train_op, 
                  global_step, 
                  train_step_kwargs)
                if should_stop:
                  tf.logging.info('Stopping Training.')
                  sv.request_stop()
                  break
              #endwhile
            except tf.errors.OutOfRangeError as e:
              tf.logging.info(
                  'Caught OutOfRangeError. Stopping Training. %s', e)
            #endtry
  
            if FLAGS.train_dir:
              tf.logging.info('Finished training! Saving model to disk.')
              sv.saver.save(sess, 
                sv.save_path, 
                global_step=sv.global_step)
              sv.stop(
                threads,
                close_summary_writer = True,
                ignore_live_threads = False)
          #endwith
        except tf.errors.AbortedError:
          tf.logging.info('Retrying training!')
          should_retry = True
  #enddef
#endclass
def main(_):
  pass

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  from gender_data import GenderTrainData 
  os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
  g_train_data = GenderTrainData(num_classes=6)
  g_train_data.load_data("annotation_train.txt", 
      "images/") #model.
  Train(g_train_data).start()

