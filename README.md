# visactivation

Written by YudeWang

A simple visualization tool for tensor activation in CNN.

### Install
```
pip install visactivation
```


### Document

```
visactivation.Tensor2Color(tensor, input_type=None, image=None, image_weight=0.3, colormap=cv2.COLORMAP_JET, act_type='max', norm_type='all')
```
Coloring the feature map in CNN to visualize the corresponding activation intensity. 

Parameters:

- tensor (numpy.ndarray) - the input tensor for visualization
- input_type (str) - 'NCHW', 'NHW','CHW','HW'. When tensor.ndim == 3, input_type must be given.
- image (numpy.ndarray, optional) - corresponding image with size NHW3 or HW3
- image_weight (float, optional) - weight of image when visualization activation
- colormap (int, str)
  - int - cv2.COLORMAP_xxx can be used here
  - str - 'voc' PASCAL VOC colormap, 'random' Random colormap
- act_type (str) - 'sum', 'max', 'mean', 'none'.
  - 'sum' - choose the sum value in channel dimension for each spatial pixel
  - 'max' - choose the max value in channel dimension for each spatial pixel
  - 'mean' - choose the mean value in channel dimension for each spatial pixel
  - 'none' - preseve the activation of C channels and visualize them independently.
- norm_type (str) - 'relu','all'.
  - 'relu' - tensor[tensor<0]=0, tensor/max(tensor)
  - 'all' - (tensor-min)/(max-min)

Return:

N x C x H x W x 3 size numpy ndarray

```
visactivation.Prob2Color(tensor, input_type=None, image=None, image_weight=0.3, colormap=cv2.COLORMAP_JET, act_type='max')
```
Coloring the probability map in CNN to visualize the corresponding activation intensity.

Parameters:

- tensor (numpy.ndarray) - the input tensor for visualization, the value should in range [0,1]
- input_type (str) - 'NCHW', 'NHW','CHW','HW'. When tensor.ndim == 3, input_type must be given.
- image (numpy.ndarray, optional) - corresponding image with size NHW3 or HW3
- image_weight (float, optional) - weight of image when visualization activation
- colormap (int, str)
  - int - cv2.COLORMAP_xxx can be used here
  - str - 'voc' PASCAL VOC colormap, 'random' Random colormap
- act_type (str) - 'sum', 'max', 'mean', 'none'.
  - 'sum' - choose the sum value in channel dimension for each spatial pixel. The result larger than 1 is cut off to 1.
  - 'max' - choose the max value in channel dimension for each spatial pixel
  - 'mean' - choose the mean value in channel dimension for each spatial pixel
  - 'none' - preseve the activation of C channels and visualize them independently.

Return:

N x C x H x W x 3 size numpy ndarray

```
visactivation.Label2Color(tensor, image=None, image_weight=0.3, colormap='random')
```
Coloring the label map predicted by to visualize the corresponding activation intensity.

Parameters:

- tensor (numpy.ndarray) - the input label for visualization, the value should in be positive integer in [0, 255].
- image (numpy.ndarray, optional) - corresponding image with size NHW3 or HW3
- image_weight (float, optional) - weight of image when visualization activation
- colormap (str) - 'voc' PASCAL VOC colormap, 'random' Random colormap

Return:

N x H x W x 3 size numpy ndarray
