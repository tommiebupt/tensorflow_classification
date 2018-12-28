import tensorflow as tf
import numpy as np
from image_processing import preprocess_image
from infer_graph import InferGraph 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

def build_runner_graph(train_dir, num_classes, image_size):
	name = tf.placeholder(dtype=tf.string, shape=(None,), name="image_name")
	#id = tf.placeholder(dtype=tf.int32, shape=(None,), name="image_id")

	image_buf = tf.placeholder(dtype=tf.string, shape=(None,), name="image_buf")
	def decode_jpeg(buf):
		return tf.image.decode_jpeg(buf,channels=3)
	image_decoded =  tf.map_fn(decode_jpeg, image_buf, dtype=tf.uint8, name="image_decoded")
	def preprocess(image):
		return preprocess_image(image, image_size, is_training=False)
	image_processed = tf.map_fn(preprocess, image_decoded, dtype=tf.float32)


	ig = InferGraph(image_processed, num_classes)
	probabilities, predictions = ig.infer_ops()
	variables_to_restore = ig.get_variables_to_restore()
	#print(tf.contrib.framework.get_model_variables())
	if probabilities is None or predictions is None or variables_to_restore is None:
			raise ValueError('results of inference cannot be None.')

	graph = tf.get_default_graph()
	with tf.name_scope('init_ops'):
		init_op = tf.global_variables_initializer()
			
		#1-D tensor: names of uninitialized variables.
		#ready_op = tf.report_uninitialized_variables()

		local_init_op = tf.group(
			tf.local_variables_initializer(), #ops: initialize all local variables. 
			tf.tables_initializer()	#ops:initialize all tables. NoOp if nonexists.
		)

	saver = tf.train.Saver(variables_to_restore)
	sess = tf.Session()
	sess.run(tf.group([init_op, local_init_op]))
	saver.restore(sess, 
		tf.train.latest_checkpoint(train_dir))
		#'./output/tfmodel/model.ckpt-77624')
	return sess, probabilities, predictions, image_buf, image_decoded, name#, id
#enddef

def export_graph(pb_path, train_dir, num_classes, image_size):
	sess, probabilities, predictions, image_buf, image_decoded, name = \
			build_runner_graph(train_dir, num_classes, image_size)
	#print(tf.get_default_graph().get_tensor_by_name(probabilities.name))
	#print(tf.get_default_graph().get_operations())

	#write pb.
	constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, 
        					[probabilities.op.name, predictions.op.name]) #output op names.
	#print(constant_graph)
	#print(sess.graph_def)
	with tf.gfile.FastGFile(pb_path, mode='wb') as f:
		f.write(constant_graph.SerializeToString())
	sess.close()
#enddef

def import_graph(pb_path, image_path):
	if not tf.gfile.Exists(image_path):
		raise ValueError("can not find image at, %s"%image_path)
	image = tf.gfile.FastGFile(image_path, 'rb').read()
	
	#read pb.
	output_graph_def = tf.GraphDef()
	with open(pb_path, 'rb') as f:
		output_graph_def.ParseFromString(f.read())
	export_tensor_names = ["image_buf:0",
			"image_decoded/TensorArrayStack/TensorArrayGatherV3:0",
			"predictions:0",
			"probabilities:0"]
	image_buf, image_decoded, predictions, probabilities =\
			 tf.import_graph_def(output_graph_def, 
				return_elements=export_tensor_names, name='')
	#print(tf.get_default_graph().get_operations())	
	with tf.Session() as sess:
		np_probs, np_preds = sess.run([probabilities, predictions], 
				feed_dict={image_buf:np.expand_dims(image,0)}) 
	print(np.hstack((np.expand_dims(np_preds, -1), np_probs)))
#enddef

def export_graph_signature(pb_path, train_dir, num_classes, width, height):
	sess, probabilities, predictions, image_buf, image_decoded, name_in, id_in = \
			build_runner_graph(train_dir, num_classes, width, height)
	name_out = tf.identity(name_in)
	id_out = tf.identity(id_in)

	builder = tf.saved_model.builder.SavedModelBuilder(pb_path)

      	class_output_tensor_info = tf.saved_model.utils.build_tensor_info(predictions)
      	score_output_tensor_info = tf.saved_model.utils.build_tensor_info(probabilities)

      	image_buf_input_tensor_info = tf.saved_model.utils.build_tensor_info(image_buf)
      	image_input_tensor_info = tf.saved_model.utils.build_tensor_info(image_decoded)
      	name_input_tensor_info = tf.saved_model.utils.build_tensor_info(name_in)
      	name_output_tensor_info = tf.saved_model.utils.build_tensor_info(name_out)
      	id_input_tensor_info = tf.saved_model.utils.build_tensor_info(id_in)
      	id_output_tensor_info = tf.saved_model.utils.build_tensor_info(id_out)

      	image_buf_prediction_signature = (
          	tf.saved_model.signature_def_utils.build_signature_def(
              		inputs={'image_buf': image_buf_input_tensor_info,
				'name': name_input_tensor_info,
				'id': id_input_tensor_info,
			},
              		outputs={
                  		'class': class_output_tensor_info,
                  		'score': score_output_tensor_info,
				'name': name_output_tensor_info,
				'id': id_output_tensor_info,
              		},
              		method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
          	)
	)

      	image_prediction_signature = (
          	tf.saved_model.signature_def_utils.build_signature_def(
              		inputs={'image': image_input_tensor_info,
				'name': name_input_tensor_info,
				'id': id_input_tensor_info,
			},
              		outputs={
                  		'class': class_output_tensor_info,
                  		'score': score_output_tensor_info,
				'name': name_output_tensor_info,
				'id': id_output_tensor_info,
              		},
              		method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
          	)
	)
    
 
      	builder.add_meta_graph_and_variables(
       		sess, 
          	[tf.saved_model.tag_constants.SERVING], 
          	signature_def_map={
              		'image_buf_predict': image_buf_prediction_signature,
              		'image_predict': image_prediction_signature,
          	}
	)	


      	builder.save()
      	sess.close()
#enddef

def import_graph_signature(pb_path, image_path):
	if not tf.gfile.Exists(image_path):
		raise ValueError("can not find image at, %s"%image_path)
	image = tf.gfile.FastGFile(image_path, 'rb').read()
    	with tf.Session(graph=tf.Graph()) as sess:
		meta_graph_def = tf.saved_model.loader.load(sess, 
			[tf.saved_model.tag_constants.SERVING], pb_path)
		signature = meta_graph_def.signature_def
		image_buf_tensor_name = signature["image_buf_predict"].inputs["image_buf"].name
		name_in_tensor_name = signature["image_buf_predict"].inputs["name"].name
		id_in_tensor_name = signature["image_buf_predict"].inputs["id"].name
		score_tensor_name = signature["image_buf_predict"].outputs["score"].name
		class_tensor_name = signature["image_buf_predict"].outputs["class"].name
		name_out_tensor_name = signature["image_buf_predict"].outputs["name"].name
		id_out_tensor_name = signature["image_buf_predict"].outputs["id"].name

		t_image_buf = sess.graph.get_tensor_by_name(image_buf_tensor_name) 
		t_name_in = sess.graph.get_tensor_by_name(name_in_tensor_name) 
		t_id_in = sess.graph.get_tensor_by_name(id_in_tensor_name) 
		t_class = sess.graph.get_tensor_by_name(class_tensor_name) 
		t_score = sess.graph.get_tensor_by_name(score_tensor_name) 
		t_name_out = sess.graph.get_tensor_by_name(name_out_tensor_name) 
		t_id_out = sess.graph.get_tensor_by_name(id_out_tensor_name) 
 
		np_id_out, np_name_out, np_probs, np_preds =\
			 sess.run([t_id_out, t_name_out, t_score, t_class], 
				feed_dict={
					t_image_buf:np.expand_dims(image,0), 
					t_name_in:np.expand_dims(np.array("helloworld"),0),
					t_id_in:np.expand_dims(np.array(11111),0),
				}) 
	print(np.hstack((np.expand_dims(np_preds, -1), np_probs, np.expand_dims(np_id_out,-1))))
#enddef

def test( train_dir, image_path, num_classes, width, height):
	if not tf.gfile.Exists(image_path):
		raise ValueError("can not find image at, %s"%image_path)
	image = tf.gfile.FastGFile(image_path, 'rb').read()

	sess, probabilities, predictions, image_buf, image_decoded, name, id = \
			build_runner_graph(train_dir, num_classes, width, height)
	#print(sess.graph_def)	
	np_probs, np_predictions = sess.run([probabilities, predictions], 
		feed_dict={image_buf:np.expand_dims(image,0)})
	print(np.hstack((np.expand_dims(np_predictions, -1), np_probs)))
	sess.close()
#enddef

if __name__ == "__main__":
	#test("./output/tfmodel", "../../data/jd_image/test/5a54d7dbN4b6d527e.jpg", 2, 256, 256)	
	export_graph("./output/pb/model.pb", "./output/tfmodel", 6, 256)	
	#import_graph("./output/pb/model.pb", "../../data/jd_image/test/5a54d7dbN4b6d527e.jpg")	
	#export_graph_signature("./output/pb", "./output/tfmodel", 2, 256, 256)	
	#import_graph_signature("./output/pb", "../../data/jd_image/test/5a54d7dbN4b6d527e.jpg")	
