from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import math
import tensorflow as tf
from eval_graph import EvalGraph 
slim = tf.contrib.slim
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', os.path.abspath('./output/tfmodel'),
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('eval_dir', os.path.abspath('./output/eval'),
                           """Directory where to write event logs """)
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_size', 256,
                            """Provide square images of this size.""")

class EvalRunner(object):
  def __init__(self, dataset, 
    eval_interval_secs = 0,
    max_num_of_evaluations = None, 
    per_process_gpu_memory_fraction=1.0):

    self._num_samples = dataset.get_num_samples()
    self._eval_interval_secs = eval_interval_secs 
    self._max_num_of_evaluations = max_num_of_evaluations

    self._session_config = tf.ConfigProto()
    self._session_config.gpu_options.allow_growth = True
    self._session_config.gpu_options.per_process_gpu_memory_fraction = \
             per_process_gpu_memory_fraction 

    self._eval_op, self._final_op, self._variables_to_restore =\
         EvalGraph(dataset).get_eval_ops() 
  #enddef

  def start(self):
    if self._eval_op is None or self._final_op is None or self._variables_to_restore is None:
      raise ValueError('eval_op or final_op or variables_to_restore cannot be None.')

    num_batches = math.ceil(self._num_samples/float(FLAGS.batch_size))-1
    print("will run %d batches."%num_batches)

    graph = tf.get_default_graph()
    with graph.as_default():
      confusion_matrix = slim.evaluation.evaluate_once(
                master="",
                checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir),
                logdir=FLAGS.eval_dir,
                num_evals=num_batches,
          session_config = self._session_config,
          final_op = self._final_op,
                eval_op=self._eval_op,
                variables_to_restore=self._variables_to_restore)
      """
      confusion_matrix = slim.evaluation.evaluation_loop(
                master="",
                checkpoint_dir = FLAGS.train_dir,
                logdir=FLAGS.eval_dir,
                num_evals=num_batches,
          session_config = self._session_config,
          final_op = self._final_op,
          max_number_of_evaluations = self._max_num_of_evaluations,
                eval_op=self._eval_op,
                eval_interval_secs=self._eval_interval_secs,
                variables_to_restore=self._variables_to_restore)
      """
      print(confusion_matrix)
  #enddef
#endclass

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  from gender_data import GenderEvalData 
  g_eval_data = GenderEvalData(num_classes=6)
  g_eval_data.load_data("annotation_test.txt", 
      "images/")
  EvalRunner(g_eval_data).start()
