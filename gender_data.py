import os
import numpy as np
import tensorflow as tf
from dataset import Dataset
import random

class GenderTrainData(Dataset):
  def __init__(self, num_classes=None, 
      num_samples=None,
      label_info=None,
      image_dir=None,
      path_pattern=None):
    super(GenderTrainData, self).__init__(num_classes, 
      num_samples,
      label_info,
      image_dir,
      path_pattern)
  #enddef

  def load_data(self, anno_file, data_dir):
    stats_dict = {}
    image_labels = []
    for line in file(anno_file):
      fields = line.strip().split("\t")
      image_path, label = fields[0], fields[1]
      impath = os.path.join(data_dir, image_path)
      if os.path.exists(impath) and image_path.endswith(".jpg"):
        image_labels.append((impath, label))
    self._num_samples = len(image_labels)
    random.shuffle(image_labels)
    self._image_dir = data_dir
    self._image_paths = np.array(image_labels)[:, 0]
    self._labels = np.array(image_labels)[:, 1].astype(int)
    print("load num samples:%d from %s"%(self._num_samples, data_dir))
  #enddef
#endclass

class GenderEvalData(Dataset):
  def __init__(self, num_classes=None, 
      num_samples=None,
      label_info=None,
      image_dir=None,
      path_pattern=None):
    super(GenderEvalData, self).__init__(num_classes, 
      num_samples,
      label_info,
      image_dir,
      path_pattern)

  def load_data(self, anno_file, data_dir, third_cate_list=None):
    stats_dict = {}
    labels = []
    image_labels = []
    for line in file(anno_file):
      fields = line.strip().split("\t")
      image_path, label = fields[0], fields[1]
      image_name = image_path.strip().split("/")[-1]
      impath = os.path.join(data_dir, image_name)
      if os.path.exists(impath) and image_name.endswith(".jpg"):
        labels.append(int(label))
        image_labels.append(os.path.join(data_dir, image_name))

    self._num_samples = len(image_labels)
    self._image_dir = data_dir
    self._image_paths = np.array(image_labels)
    self._labels = np.array(labels)
    print("num samples:%d, female:%d"%(self._num_samples,sum(labels)))
  #enddef
#endclass

class GenderTestData(Dataset):
  def __init__(self, num_classes=None, 
      num_samples=None,
      label_info=None,
      image_dir=None,
      path_pattern=None):
    super(GenderTestData, self).__init__(num_classes, 
      num_samples,
      label_info,
      image_dir,
      path_pattern)

  def load_data(self, image_dir, path_pattern):
    self._path_pattern = path_pattern
    self._image_dir = image_dir
    self._num_samples = len(tf.gfile.Glob(path_pattern)) 
    print("num samples:%d"%self._num_samples)
  #enddef
#endclass
