import tensorflow as tf
from input_graph import * 
from preprocess_graph import preprocess_image
from infer_graph import InferGraph 

class XEvalGraph(object):
  def __init__(self, dataset, height, width, batch_size=32):
    self._dataset = dataset
    self._height = height
    self._width = width
    self._batch_size = batch_size
  #enddef

  def get_eval_ops(self):
    image_paths, labels = self._dataset.get_image_paths()

    eval_data= PathsInputforEval()  
    filename_batch, image_batch, label_batch = eval_data.get_batches(image_paths, 
        labels, 
        preprocess_image, 
        self._height,
        self._width,
        self._batch_size)
        
    ig =  InferGraph(image_batch, 
        self._dataset.get_num_classes())
    probs, predictions = ig.infer_ops()
    variables_to_restore = ig.get_variables_to_restore()
          return filename_batch, label_batch, probs, predictions, variables_to_restore
  #enddef
#endclass

def test():
  from gender_data import GenderEvalData 
  os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
  g_eval_data = GenderEvalData(num_classes=2, label_info={'male':0, 'female':1})
  g_eval_data.load_data("../data/test_samples.dat", 
      "../data/test")

  _, _label_batch, _probs, _predictions, _variables = XEvalGraph(g_eval_data, 256, 256).get_eval_ops()  
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
      np_label_batch, np_probs, _ = sess.run([_label_batch, _probs, _predictions])
      print(np_probs)
    except tf.errors.OutOfRangeError:
      print("done")
    finally:
      coord.request_stop()
    coord.join(threads)
    
if __name__ == "__main__":
  test()
