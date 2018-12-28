import abc

class Dataset(object):
	__metaclass__ = abc.ABCMeta

	def __init__(self, num_classes=None, 
			num_samples=None,
			label_info=None,
			image_dir=None,
			path_pattern=None):
		self._num_classes = num_classes
		self._num_samples = num_samples	
		self._label_info = label_info
		self._image_dir = image_dir
		self._path_pattern = path_pattern

		self._class_name_segment_index = None
		self._image_paths = None
		self._images = None
		self._labels = None
	#enddef
	
	@abc.abstractmethod
	def load_data(self, *args, **kwargs):
		raise NotImplementedError
	#enddef
	
	def get_num_classes(self):
		return self._num_classes
	#enddef

	def get_num_samples(self):
		return self._num_samples 
	#enddef

	def get_label_info(self):
		"""return a map of class name to label.
		"""
		return self._label_info
	#enddef

	def get_image_dir(self):
		"""return directory of samples, pattern of image path, 
		index of class name segment in image path(-1 if nonexists.)
		"""
		return self._image_dir, self._path_pattern, self._class_name_segment_index
	#enddef

	def get_image_paths(self):
		"""return a ndarray of image paths 
		and a ndarray of image labels(None if nonexists).
		"""
		return self._image_paths, self._labels	
	#enddef

	def get_image_array(self):
		"""return a ndarray of image 
		and a ndarray of image labels(None if nonexists).
		"""
		return self._images, self._labels
	#enddef
#endclass
