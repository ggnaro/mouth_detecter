# Mouth Detector model

It's a computer vision model inference on human faces, to cropping the face area and then detect the 2 lips of mouth.
It would be provides the lips inside area, called as "mouthinside". <br>
![sample2](./assets/sample2.png)<br>


## Usage

Currently, provided python and inference on one image only, to modified the image path in [inference_mouthinside.py](./inference_mouthinside.py)

**envs limits: python 3.8, tensorflow 2.2.0, openvino 2021.4**


## Data
utilizing with [Microsoft Synthetics](https://github.com/microsoft/FaceSynthetics),  totally 99,773 volumes.

## Training framweork
Tensorflow keras with Unet, refer to an [introduction](https://github.com/veer2701/Image-Segmentation-with-U-Net/blob/main/6%20Image_segmentation_Unet_v2.ipynb)

## Testing results on public datasets
Need to updated

## Model conversion
Now, the version is trained done in 2021.11, and for edge inferrence, also provided openvino version.
Meanwhile, the version is too old to running on some framework nowtimes.

## ToDO
- [ ] Re-training with pytorch  
- [ ] Conversion to ONNX
- [ ] Evaluation of Public Datasets, IoU metrics

## Further
- [ ] C++, Video frames Analysis tools


## Reference
1. techniques, https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
2. techniques, https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
3. techniques, https://en.wikipedia.org/wiki/Dynamic_time_warping
4. techniques, [correlation adn convolution](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10452/104520Y/Teaching-the-concept-of-convolution-and-correlation-using-Fourier-transform/10.1117/12.2267976.full?SSO=1)
5. papers, [Effect of the output activation function on the probabilities and errors in medical image segmentation](https://arxiv.org/abs/2109.00903)
6. papers, [Determination of Hue Saturation Value (HSV)color feature in kidney histology image](https://iopscience.iop.org/article/10.1088/1742-6596/2157/1/012020/pdf)