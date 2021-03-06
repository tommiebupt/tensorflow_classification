import tensorflow as tf
from input_graph import * 
from image_processing import preprocess_image
from metric_graph import MetricGraph 

FLAGS = tf.app.flags.FLAGS

class EvalGraph(object):
  def __init__(self, dataset):
    self._dataset = dataset
  #enddef

  def get_eval_ops(self):
    image_paths, labels = self._dataset.get_image_paths()

    eval_data= PathsInputforEval()  
    _, image_batch, label_batch = eval_data.get_batches(image_paths, 
        labels, 
        preprocess_image, 
        FLAGS.image_size,
        FLAGS.batch_size)
        
    metric_g =  MetricGraph(image_batch, 
        label_batch, 
        self._dataset.get_num_classes())
    eval_op, final_op = metric_g.eval_ops()
    variables_to_restore = metric_g.get_variables_to_restore()
    return eval_op, final_op, variables_to_restore
  #enddef
#endclass

def test():
  from gender_data import GenderEvalData 
  os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
  g_eval_data = GenderEvalData(num_classes=6)
  g_eval_data.load_data("test_samples.dat", 
      "images/test")

  _eval_op, _final_op, _variables = EvalGraph(g_eval_data, 256).get_eval_ops()  
  with tf.Session() as sess:
    init_op = tf.group([tf.global_variables_initializer(),
        tf.local_variables_initializer(), tf.tables_initializer()])
    sess.run(init_op)
    saver = tf.train.Saver(_variables)
    saver.restore(sess, tf.train.latest_checkpoint("./output/tfmodel"))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    try:
      for j in range(100):
        _, confusemat = sess.run([_eval_op, _final_op])
        print(confusemat)
    except tf.errors.OutOfRangeError:
      print("done")
    finally:
      coord.request_stop()
    coord.join(threads)
    
if __name__ == "__main__":
  test()
