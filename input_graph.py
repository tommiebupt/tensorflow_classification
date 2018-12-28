from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
import abc
slim = tf.contrib.slim


class InputBase(object):
  __metaclass__ = abc.ABCMeta
  
  @abc.abstractmethod
  def _create_batches(self, *args, **kwargs):
    raise NotImplementedError
  #enddef

  def get_batches(self, *args, **kwargs):
    return self._create_batches(*args, **kwargs)
  #enddef
#endclass


class DirectoryInputforTrain(InputBase):
  def _create_batches(self, sample_dir, 
      file_path_pattern, 
      class_name_index,
      class_name_to_label,
      preprocess_fn,
      image_size,
      batch_size=32,
      num_epochs = None,
      num_threads=4):
      """
      For training.
      """
      if not tf.gfile.IsDirectory(sample_dir) or not tf.gfile.Exists(sample_dir):
        raise ValueError("can not find sample directory, %s"%sample_dir)
  
      files = tf.gfile.Glob(os.path.join(sample_dir, file_path_pattern))
      print("find %d files."%len(files))
      fileQueue = tf.train.string_input_producer(files, 
      num_epochs=num_epochs, capacity=1024+10*batch_size, shuffle=True)
      reader = tf.WholeFileReader()
      file_name, file_content = reader.read(fileQueue, name="image_buf")
      sparse_path_fields = tf.string_split(tf.expand_dims(file_name,-1), "/")
      class_name = sparse_path_fields.values[class_name_index]
      table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(class_name_to_label.keys(), 
        class_name_to_label.values()),-1) # TOADD: tf.tables_initializer().run()
      label = table.lookup(class_name)      
      image = tf.image.decode_jpeg(file_content, channels=3, name="image_decoded")
      #image = tf.image.decode_image(file_content, channels=3, name="image_decoded")
      image = preprocess_fn(image, image_size, is_training=True)
      image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size,
          num_threads=num_threads, capacity=128+3*batch_size, min_after_dequeue=128) 
      return image_batch, label_batch
  #enddef
#encclass

class NDArrayInputforTrain(InputBase):
  def _create_batches(self, images, 
      labels,
      preprocess_fn,
      image_size,
      batch_size=32, 
      num_epochs=None, 
      num_threads=4,  
      shuffle=True, 
      allow_smaller_final_batch=False):
      """
      For training.
      """
      image, label = tf.train.slice_input_producer([images, labels], 
        num_epochs=num_epochs,
        capacity=128+3*batch_size,
         shuffle=shuffle)
      #image = tf.image.decode_jpeg(file_content, channels=3, name="image_decoded")
      image = preprocess_fn(image, image_size, is_training=True)
      image_batch, label_batch = tf.train.batch([image,label], 
        batch_size=batch_size,
        num_threads = num_threads,
        allow_smaller_final_batch=allow_smaller_batch,
        capacity=128+3*batch_size)
      return image_batch, label_batch
  #enddef
#endclass


class PathsInputforTrain(InputBase):
  def _create_batches(self, image_paths, 
      labels, 
      preprocess_fn,
      image_size,
      batch_size=32,
      num_threads=4,  
      num_epochs=None, 
      shuffle=True, 
      allow_smaller_final_batch=False):
      """
      For training.
      """
    
      image_path, label = tf.train.slice_input_producer([image_paths, labels], 
      num_epochs=num_epochs, capacity=1024+10*batch_size, shuffle=shuffle)
      #reader = tf.WholeFileReader()
      file_content = tf.read_file(image_path, name="image_buf") #reader.read(image_path)
      image = tf.image.decode_jpeg(file_content, channels=3, name="image_decoded")
      image = preprocess_fn(image, image_size, is_training=True)
      image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size,
          num_threads=num_threads, capacity=512+3*batch_size, 
          allow_smaller_final_batch=allow_smaller_final_batch)
      return image_batch, label_batch
  #enddef 
#endclass


class DirectoryInputforEval(InputBase):
  def _create_batches(self, sample_dir, 
      file_path_pattern, 
      class_name_index, 
      class_name_to_label,
      preprocess_fn,
      image_size,
      batch_size=32):
      """
      For evaluating or testing, run once orderly.
      """
      if not tf.gfile.IsDirectory(sample_dir) or not tf.gfile.Exists(sample_dir):
        raise ValueError("can not find sample directory, %s"%sample_dir)
  
      files = tf.gfile.Glob(os.path.join(sample_dir, file_path_pattern))
      print("Find %d files."%len(files))
      fileQueue = tf.train.string_input_producer(files, 
      num_epochs=1, capacity=1024+10*batch_size, shuffle=False)
      reader = tf.WholeFileReader()
      file_name, file_content = reader.read(fileQueue)
      sparse_path_fields = tf.string_split(tf.expand_dims(file_name,-1), "/")
      class_name = sparse_path_fields.values[class_name_index]
      table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(class_name_to_label.keys(), 
        class_name_to_label.values()),-1) # TOADD: tf.tables_initializer().run()
      label = table.lookup(class_name)      
      image = tf.image.decode_jpeg(file_content, channels=3)
      image = preprocess_fn(image, image_size, is_training=False)
      image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size,
          num_threads=1, capacity=128+3*batch_size, 
          allow_smaller_final_batch=True) 
      return image_batch, label_batch
  #enddef
#endclass

class NDArrayInputforEval(InputBase):
  def _create_batches(self, images, 
        labels,
        preprocess_fn,
        image_size,
        batch_size=32): 
      """
      For evaluating or testing, run once orderly.
      """
      image, label = tf.train.slice_input_producer([images, labels], 
        num_epochs=1,
        capacity=128+3*batch_size,
         shuffle=False)
      #image = tf.image.decode_jpeg(file_content, channels=3)
      image = preprocess_fn(image, image_size, is_training=False)
      image_batch, label_batch = tf.train.batch([image, label], 
        batch_size=batch_size,
        num_threads = 1,
        allow_smaller_final_batch=True,
        capacity=128+3*batch_size)
      return image_batch, label_batch
  #enddef
#endclass

class PathsInputforEval(InputBase):
  def _create_batches(self, image_paths, 
      labels, 
      preprocess_fn,
      image_size,
      batch_size=32): 
      """
    For evaluating or testing, run once orderly.
      """
    
      image_path, label = tf.train.slice_input_producer([image_paths, labels], 
      num_epochs=1, capacity=1024+10*batch_size, shuffle=False)
      file_content = tf.read_file(image_path)
      image = tf.image.decode_jpeg(file_content, channels=3)
      image = preprocess_fn(image, image_size, is_training=False)
      filename_batch, image_batch, label_batch =\
       tf.train.batch([image_path, image, label], batch_size=batch_size,
          num_threads=1, capacity=128+3*batch_size, 
          allow_smaller_final_batch=True) 

      return filename_batch, image_batch, label_batch
  #enddef
#endclass

class DirectoryInputforTest(InputBase):
  def _create_batches(self,  image_dir,
          file_path_pattern,
          preprocess_fn,
          image_size,
          batch_size=32): 

      if not tf.gfile.IsDirectory(image_dir) or not tf.gfile.Exists(image_dir):
        raise ValueError("can not find sample directory, %s"%image_dir)
  
      files = tf.gfile.Glob(os.path.join(image_dir, file_path_pattern))
      fileQueue = tf.train.string_input_producer(files, 
      num_epochs=1, capacity=1024+10*batch_size, shuffle=False)
      reader = tf.WholeFileReader()
      file_name, file_content = reader.read(fileQueue)
      image = tf.image.decode_jpeg(file_content, channels=3)
      image = preprocess_fn(image, image_size, is_training=False)
      file_name_batch, image_batch = tf.train.batch([file_name, image], batch_size=batch_size,
          num_threads=1, capacity=128+3*batch_size, 
          allow_smaller_final_batch=True) 
      return file_name_batch, image_batch
  #enddef
#endclass

def test():
  os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
  def default_preprocess_fn(image, image_size, is_training=False):
      return tf.image.resize_images(image, size=(image_size,image_size))

  image_batch, label_batch = DirectoryInputforTrain().get_batches(
          "/export/data0/user/taoriming/data/jd_image/", 
          "*/*.jpg", 
          -2, 
          {"male":0,"female":1},
          default_preprocess_fn, 
          256, 
          256)
  with tf.Session() as sess:
    init_op = tf.group([tf.local_variables_initializer(), tf.tables_initializer()])
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    try:
      for j in range(100):
            np_image_batch, np_label_batch = sess.run([image_batch,label_batch])
            print(np_image_batch.shape,np_label_batch)
    except tf.errors.OutOfRangeError:
      print("done")
    finally:
      coord.request_stop()
    coord.join(threads)
#enddef

if __name__ == "__main__":
  test()
