import tensorflow as tf
import pickle
import shutil
import os
from visualize_model import *
#from xeval_runner import XEvalRunner

def extract_tensors_by_meta(tensor_names, image_path, model_ckpt_path):
  if not tf.gfile.Exists(image_path):
    raise ValueError("can not find image at, %s"%image_path)
  image = tf.gfile.FastGFile(image_path, 'rb').read()
    
  tensor_dict = {}
  tensor_value_dict = {}
  loaded_graph = tf.Graph()
  with tf.Session(graph=loaded_graph) as sess:
    # Load model
    loader = tf.train.import_meta_graph(model_ckpt_path + '.meta')
    loader.restore(sess, model_ckpt_path)

    # Get Tensors from loaded model
    tensor_dict['image_buf:0'] = loaded_graph.get_tensor_by_name('image_buf:0')
    tensor_dict['image_decoded:0'] = loaded_graph.get_tensor_by_name('image_decoded:0')
    tensor_dict['predictions:0'] =  loaded_graph.get_tensor_by_name('predictions:0')
    tensor_dict['probabilities:0'] = loaded_graph.get_tensor_by_name('probabilities:0')
  
    for tensor_name in tensor_names:
      tensor_dict[tensor_name] = loaded_graph.get_tensor_by_name(tensor_name)  
    
    tensor_value_dict = sess.run(tensor_dict, feed_dict={tensor_dict['image_buf:0']:image})
  return tensor_value_dict
#enddef

def extract_tensors_by_pb(tensor_names, image_path, model_pb_path):
  if not tf.gfile.Exists(image_path):
    raise ValueError("can not find image at, %s"%image_path)
  image = tf.gfile.FastGFile(image_path, 'rb').read()

  #read pb.
  output_graph_def = tf.GraphDef()
  with open(model_pb_path, 'rb') as f:
    output_graph_def.ParseFromString(f.read())

    image_buf, image_decoded, predictions, probabilities =\
       tf.import_graph_def(output_graph_def, 
        return_elements=[
          "image_buf:0", 
          "image_decoded/TensorArrayStack/TensorArrayGatherV3:0", 
          "predictions:0", 
          "probabilities:0"],
        name='')
  
  #print(tf.get_default_graph().get_operations())
  tensor_dict = {}  
  tensor_value_dict = {}
  tensor_dict['image_buf:0'] = image_buf 
  tensor_dict['image_decoded:0'] = image_decoded
  tensor_dict['predictions:0'] = predictions
  tensor_dict['probabilities:0'] = probabilities
  
  with tf.Session() as sess:
    for tensor_name in tensor_names:
      tensor_dict[tensor_name] = sess.graph.get_tensor_by_name(tensor_name)  
    tensor_value_dict = sess.run(tensor_dict, {image_buf:np.expand_dims(image, 0)}) 
  tf.reset_default_graph()
  return tensor_value_dict 
#enddef

def cam_maps(image_path, model_path):
  tensor_value_dict = extract_tensors_by_pb(['resnet_v1_50/logits/weights:0',
      'resnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0'],
      image_path,
      model_path)  
  
  image = tensor_value_dict['image_decoded:0']
  activations = tensor_value_dict['resnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0']
  weights = tensor_value_dict['resnet_v1_50/logits/weights:0']
  predictions = tensor_value_dict['predictions:0']
  probabilities = tensor_value_dict['probabilities:0']
  print(probabilities)
  print(image.shape, activations.shape, weights.shape)
  
  labels = predictions.tolist()
  saved_paths = [ 
    os.path.splitext(image_path)[0]+'_%d_%.2f'%(label, probabilities.tolist()[0][label])+'.png' \
      for label in labels
    ]
  plot_cam_map(image, activations, weights, labels, saved_paths)
#enddef

def images_cam_maps(image_dir, model_path):
  for filename in os.listdir(image_dir):
    path = os.path.join(image_dir, filename)
    if os.path.isfile(path) and path.endswith('.jpg'):  
      cam_maps(path, './output/pb/model.pb')
#enddef

def confusemat():
  with open('./output/pkl/eval_results.pkl', 'rb') as fr:
    eval_results = pickle.load(fr)
  cfm_true_labels = []
  cfm_pred_labels = []
  for k,v in eval_results.items():
    _, true_labels, pred_labels, _= zip(*v)
    plot_normal_confusion_matrix(list(true_labels), 
      list(pred_labels), ['male', 'female'], k.split("/")[-1])
#enddef

def roc():
  with open('./output/pkl/eval_results.pkl', 'rb') as fr:
    eval_results = pickle.load(fr)
  roc_true_labels = []
  roc_pred_scores = []
  roc_descs = []
  for k,v in eval_results.items():
    _, true_labels, _, scores = zip(*v)
    roc_true_labels.append(list(true_labels))
    roc_pred_scores.append(list(scores))
    roc_descs.append("%s"%k)
  plot_roc(np.array(roc_true_labels), np.array(roc_pred_scores), np.array(roc_descs), "Gender-Model")
#enddef

def prc():
  with open('./output/pkl/eval_results.pkl', 'rb') as fr:
    eval_results = pickle.load(fr)
  roc_true_labels = []
  roc_pred_scores = []
  roc_descs = []
  for k,v in eval_results.items():
    _, true_labels, _, scores = zip(*v)
    roc_true_labels.append(list(true_labels))
    roc_pred_scores.append(list(scores))
    roc_descs.append("%s"%k)
  plot_prc(np.array(roc_true_labels), np.array(roc_pred_scores), np.array(roc_descs), "Gender-Model")
#enddef

def false_prediction():
  with open('./output/pkl/eval_results.pkl', 'rb') as fr:
    eval_results = pickle.load(fr)
  for k,v in eval_results.items():
    filenames, true_labels, pred_labels, scores = zip (*v)
    np_filenames = np.array(filenames)
    np_true_labels = np.array(true_labels)
    np_pred_labels = np.array(pred_labels)
    np_scores = np.array(scores)

    a = np_filenames[np_true_labels != np_pred_labels].tolist()
    b = np_scores[np_true_labels != np_pred_labels].tolist() 
    print(zip(a,b))
    
    for image_path in a:
      shutil.copy(image_path, "./images/"+image_path.split("/")[-1])      
#enddef

def test_extract_tensors():
  tensor_value_dict = extract_tensors_by_pb(['resnet_v1_50/logits/BiasAdd:0', 
      'resnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0'],
      "./test/5a54d7dbN4b6d527e.jpg",
      "./output/pb/model.pb")  
  
  # NOTE: thread suspended!
  #tensor_value_dict = extract_tensors_by_meta(['resnet_v1_50/logits/BiasAdd:0', 
  #    'resnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0'],
  #    "../../data/jd_image/test/5a54d7dbN4b6d527e.jpg",
  #    "./output/tfmodel/model.ckpt-500003")  
  #print(tensor_value_dict) 
#enddef

def test_cam_maps():
  cam_maps( "./test/5a54d7dbN4b6d527e.jpg",
      "./output/pb/model.pb")
#enddef

def test_roc():
  roc()
#enddef
  
def test_prc():
  prc()
#enddef

def test_confusemat():
  confusemat()
#enddef

if __name__ == "__main__":
  #false_prediction()
  images_cam_maps('./images', './output/pb/model.pb')
