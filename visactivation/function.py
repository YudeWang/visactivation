import cv2
import numpy as np

def _voc_colormap(label):
	# label: HW
	# return: HW3
	m = label.astype(np.uint8)
	r,c = m.shape
	cmap = np.zeros((r,c,3), dtype=np.uint8)
	cmap[:,:,0] = (m&1)<<7 | (m&8)<<3
	cmap[:,:,1] = (m&2)<<6 | (m&16)<<2
	cmap[:,:,2] = (m&4)<<5
	cmap[m==255] = [255,255,255]
	return cmap

def _voc_colorarray(idx):
	# idx: int
	# return: array
	idx = int(idx)
	colorarray = np.array([(idx&1)<<7|(idx&8)<<3, (idx&2)<<6|(idx&16)<<2, (idx&4)<<5])
	return colorarray

def _rand_colormap(label):
	h,w = label.shape
	cmap = np.zeros((h,w,3), dtype=np.uint8)
	color = (np.random.random((256,3))*0.6+0.4)*255
	color = color.astype(np.uint8)
	cmap = color[label]
	return cmap

def _rand_colorarray():
	colorarray = (np.random.random((3))*0.6+0.4)*255
	colorarray = colorarray.astype(np.uint8)
	return colorarray

def _check(tensor, input_type=None, image=None):
	# input_type: HW, CHW, NHW, NCHW
	# image: 3HW, HW3, NHW3, N3HW
	# return: NCHW tensor/ NHW3 image
	if input_type == 'HW':
		assert tensor.ndim == 2
		tensor = np.expand_dims(tensor, axis=(0,1))
	elif input_type == 'CHW':
		assert tensor.ndim == 3
		tensor = np.expand_dims(tensor, axis=0)
	elif input_type == 'NHW':
		assert tensor.ndim == 3
		tensor = np.expand_dims(tensor, axis=1)
	elif input_type == 'NCHW':
		assert tensor.ndim == 4
	else:
		raise ValueError('input_type is not in [HW, CHW, NHW, NCHW]')
	if image is not None:
		assert image.ndim ==3 or image.ndim == 4
		if image.ndim == 3:
			if image.shape[0] == 3:
				image = np.expand_dims(image.transpose(1,2,0), axis=0)
			else:
				image = np.expand_dims(image, axis=0)
		else:
			if image.shape[1] == 3:
				image = image.transpose(0,2,3,1)
		assert image.shape[3] == 3
	return tensor, image
	

def _label2color_single(label, colormap='voc'):
	# label: HW
	# return: HW3
	h,w = label.shape
	if colormap == 'voc':
		label_color = _voc_colormap(label)
	elif colormap == 'random':
		label_color = _rand_colormap(label)
	else:
		raise ValueError('colormap=%s is not supported'%colormap)
	return label_color

def _label2color_multi(label, colormap='voc'):
	# label: N1HW
	# return: NHW3
	n, c, h, w = label.shape
	label_color = np.zeros((n,h,w,3),dtype=np.uint8)
	for i in range(n):
		label_color[i] = _label2color_single(label[i,0], colormap)
	return label_color

def Label2Color(tensor, image=None, image_weight=0.3, colormap='random'):
	# tensor: HW, NHW, N1HW
	# image: 3HW, HW3, NHW3, N3HW
	# input_type: HW, CHW, NHW, NCHW
	# return: NHW3
	if tensor.ndim == 2:
		label, image = _check(tensor, 'HW', image)
	elif tensor.ndim == 3:
		label, image = _check(tensor, 'NHW', image)
	elif tensor.ndim == 4 and tensor.shape[1] == 1:
		label, image = _check(tensor, 'NCHW', image)
	else:
		raise ValueError('tensor.ndim=%d is not supported'%tensor.ndim)
	label = label.astype(np.uint8)
	label_color =  _label2color_multi(label, colormap)
	if image is not None:
		for i in range(label_color.shape[0]):
			label_color[i] = cv2.addWeighted(image[i], image_weight, label_color[i], (1-image_weight), 0)
	return label_color
	
def _prob2color_singlechannel(prob, colormap=cv2.COLORMAP_JET):
	# prob: NHW
	# colormap: array, int
	# return: NHW3
	n, h, w = prob.shape
	prob_255 = (prob*255).astype(np.uint8)
	prob_255 = np.expand_dims(prob_255, axis=-1)
	color = np.zeros((n,h,w,3))
	if isinstance(colormap, np.ndarray):
		c = np.array(colormap).reshape((1,1,1,-1)).astype(np.float32)/255
		color = (prob_255*c)
	elif isinstance(colormap, int):
		for i in range(n):
			color[i] = cv2.cvtColor(cv2.applyColorMap(prob_255[i], colormap), cv2.COLOR_BGR2RGB)
			#color[i] = prob_255[i]
	else:	
		raise ValueError('colormap is not supported')
	return color.astype(np.uint8)

def _prob2color_multichannel(prob, colormap=cv2.COLORMAP_JET, act_type='max'):
	# prob: NCHW
	# colormap: array, int
	# act_type: 'sum', 'max', 'mean', 'none'
	# return: NCHW3
	n, c, h, w = prob.shape
	if isinstance(colormap, int):
		if act_type == 'sum':
			prob_sum = np.sum(prob, axis=1, keepdims=False)
			prob_sum[prob_sum > 1] = 1
			prob_color = _prob2color_singlechannel(prob_sum, colormap)
			prob_color = np.expand_dims(prob_color, axis=1)
		elif act_type == 'max':
			prob_max = np.max(prob, axis=1, keepdims=False)
			prob_color = _prob2color_singlechannel(prob_max, colormap)
			prob_color = np.expand_dims(prob_color, axis=1)
		elif act_type == 'mean':
			prob_mean = np.mean(prob, axis=1, keepdims=False)
			prob_color = _prob2color_singlechannel(prob_mean, colormap)
			prob_color = np.expand_dims(prob_color, axis=1)
		elif act_type == 'none':
			prob_color = np.zeros((n,c,h,w,3), dtype=np.uint8)
			for i in range(c):
				prob_color[:,i,:,:,:] = _prob2color_singlechannel(prob[:,i,:,:], colormap)
		else:
			raise ValueError('act_type is not supported')
	else:
		prob_color = np.zeros((n,c,h,w,3), dtype=np.uint8)
		for i in range(c):
			if colormap == 'voc':
				color_sub = _voc_colorarray(i)
			elif colormap == 'random':
				color_sub = _rand_colorarray()
			else:
				raise ValueError('colormap=%s is not supported'%colormap)
			prob_color[:,i,:,:,:] = _prob2color_singlechannel(prob[:,i,:,:], color_sub)
		if act_type == 'sum' or act_type == 'mean':
			prob_color = np.mean(prob_color, axis=1, keepdims=True)
		elif act_type == 'max':
			prob_max = np.max(prob, axis=1, keepdims=True)
			prob_color[prob_max!=prob] = 0
			prob_color = np.sum(prob_color, axis=1, keepdims=True)
		else:
			assert act_type == 'none'

	return prob_color.astype(np.uint8)

def Prob2Color(tensor, input_type=None, image=None, image_weight=0.3, colormap=cv2.COLORMAP_JET, act_type='max'):
	# tensor: NCHW, CHW, NHW, HW
	# colormap: array, int
	# act_type: 'sum', 'max', 'mean', 'none'
	# return: NCHW3
	if tensor.ndim == 2:
		prob, image = _check(tensor, 'HW', image)
	elif tensor.ndim == 3:
		prob, image = _check(tensor, input_type, image)
	elif tensor.ndim == 4:
		prob, image = _check(tensor, 'NCHW', image)
	else:
		raise ValueError('tensor.ndim=%d is not supported'%tensor.ndim)
	n,c,h,w = prob.shape
	prob_color =  _prob2color_multichannel(prob, colormap, act_type)
	if image is not None:
		for i in range(n):
			for j in range(prob_color.shape[1]):
				prob_color[i,j] = cv2.addWeighted(image[i], image_weight, prob_color[i,j], (1-image_weight), 0)
	return prob_color

def _relu_norm(tensor, e=1e-9):
	# tensor: NCHW
	# return: NCHW
	n,c,h,w = tensor.shape
	tensor[tensor<e] = 0
	tensor_max = np.max(tensor.reshape(n,c,-1), axis=-1).reshape(n,c,1,1)+e
	tensor = tensor/tensor_max
	return tensor
	
def _all_norm(tensor, e=1e-9):
	# tensor: NCHW
	# return: NCHW
	n,c,h,w = tensor.shape
	tensor_max = np.max(tensor.reshape(n,c,-1), axis=-1).reshape(n,c,1,1)
	tensor_min = np.min(tensor.reshape(n,c,-1), axis=-1).reshape(n,c,1,1)
	tensor = (tensor-tensor_min)/(tensor_max-tensor_min+e)
	return tensor

def Tensor2Color(tensor, input_type=None, image=None, image_weight=0.3, colormap=cv2.COLORMAP_JET, act_type='max', norm_type='all'):
	# tensor: NCHW, CHW, NHW, HW
	# colormap: array, int
	# act_type: 'sum', 'max', 'mean', 'none'
	# norm_type: 'all', 'relu', 'none'
	# return: NCHW3
	if tensor.ndim == 2:
		tensor, image = _check(tensor, 'HW', image)
	elif tensor.ndim == 3:
		tensor, image = _check(tensor, input_type, image)
	elif tensor.ndim == 4:
		tensor, image = _check(tensor, 'NCHW', image)
	else:
		raise ValueError('tensor.ndim=%d is not supported'%tensor.ndim)
	if norm_type == 'all':
		tensor = _all_norm(tensor)
	elif norm_type == 'relu':
		tensor = _relu_norm(tensor)
	else:
		tensor_max = np.max(tensor)
		tensor_min = np.min(tensor)
		assert norm_type == 'none' and tensor_max <= 1 and tensor_min >= 0
	n,c,h,w = tensor.shape
	tensor_color =  _prob2color_multichannel(tensor, colormap, act_type)
	if image is not None:
		for i in range(n):
			for j in range(tensor_color.shape[1]):
				tensor_color[i,j] = cv2.addWeighted(image[i], image_weight, tensor_color[i,j], (1-image_weight), 0)
	return tensor_color

