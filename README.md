# Semantic Segmentation

[//]: # (Image References)
[image1]: ./img/output.gif
[image2]: ./img/vgg16.png
[image3]: ./img/mobilenet_v1.png
[image3]: ./img/video_11fps.png

<p align="center">
     <img src="./img/output.gif" alt="video demo" width="100%" height="100%">
     <br>semantic segmentation with a FCN network
</p>
  
  
### Project description

Pixel wise classification is implemented via a Fully Convolutional Network (FCN) making use of a VGG16 encoder pre-trained on Imagenet for optimal performances.  
  
Two databases are used for training:   
- Kitti: http://www.cvlibs.net/datasets/kitti/eval_road.php   
- Cityscapes: https://www.cityscapes-dataset.com    

The FCN8s network is trained to perform pixel wise classification among 20 classes as per official cityscapes benchmark.  

The implementation is tested againt the official cityscapes test set in terms of IOU metric.  
Optimizations on the inference part are being done thanks to the use of Tensorflow freeze, optimize and transform_graph tools.  
cf https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md  
A demonstration video is provided: based on cityscapes video sequences for qualitative evalutaion.

The current status in terms of performance is the following:  
* Cityscapes test set IOU: 73.48% with (256, 512) input images.   
So with a subsampling of 4 compared to raw images and **ground truth** images provided by Cityscapes.   
* inference time on a GTX 1080 TI: 63 ms per image with 32 bits weights  

### How to train and test on Kitti and Cityscapes test sets

Training and testing on Kitti for binary pixel wise classification (road / not road):  

```
python main.py
```
The results on the Kitti test set are stored in runs subdirectory.  
  
Training and testing on Cityscapes for 20 classes pixel wise classification:  
```
python main_cityscapes.py --epoch 100 --lr 5e-4 --batch-size 4 --early-stop True --patience 3
```  

The results on the Cityscapes test set are stored in city_runs subdirectory.    
Moreover the mean IOU metric as per official Cityscape benchmark (with a subsampling of 4 in our case) is provided.    

### How to optimize for inference

In optimized_inference subdirectory:  
cf optimized_inference/doc/howto.txt  
The result is a protobuf file optimized for inference: stored in optimized_inference/models/optimized_graph.pb    
That will be used later on to perform video decoding.    

### How to measure inference time and test on video  

In optimized_inference subdirectory:   
```
python test_graph.py --graph models/transformed_graph.pb --video data/video.mp4
```

The result is stored as: optimized_inference/data/output.mp4    
  
  
  <p align="center">
     <img src="./img/video_11fps.png" alt="video at 11 fps" width="100%" height="100%">
     <br>semantic segmentation at 11 fps
</p>


### Network architecture  

A VGG16 encoder pre-trained on Imagenet is being used with the following modifications:
- input image is systematically downscaled to (256, 512). So by a factor of 4 compared to Cityscapes raw images in order to reduce computation time 
- the fully connected layers are replaced by 1x1 convolutions

 <p align="center">
     <img src="./img/vgg16.png" alt="vgg16" width="75%" height="75%">
     <br>vgg16
</p>

The decoder is in charge of upsampling back to the original image size (i.e. (256, 512) in our case) via 3 consecutive conv2d_transpose operations (x2 x2 x8 => x32 in total; as the input image was downscaled in the encoder by a factor of 32).  
Skip layers are being used to retain and propagate information that was present in the encoder before fully downsampling the image and that would otherwise be lost: this increases the accuracy of the semantic segmenter. 

Concerning the VGG16 encoder:  
It was designed in 2014 by Visual Geometry Group at Oxford university and achieved best results at Imagenet classification competition. VGG has a simple and elegant architecture which makes it great for transfer learning: the VGG architecture is just a long sequence of 3x3 convolutions broken up by 2x2 pooling layers and finished by 3 fully connected layers at the end. Lots of engineers use VGG as a starting point for working on other images deep learning tasks and it works really well. The flexibility of VGG is one of its great strength. Nevertheless it is quite old now and pretty big.  
This very big front end could be replaced by other alternatives like GoogleNet or MobileNet, also pre-trained on Imagenet, if we want to increase fps. The trade off in terms of complexity vs accuracy is depicted below. MobileNet is especially recommended for embedded devices applications.

 <p align="center">
     <img src="./img/mobilenet_v1.png" alt="vgg16" width="50%" height="50%">
     <br>MobileNet
</p>

Note that there exists also netwrok architectures like ENet and ERFNet targetting specifically low power embedded devices and enabling real-time semantic segmentation as well. Based on Cityscapes benchmark results, ERFNet looks very interesting: https://www.cityscapes-dataset.com/benchmarks/#pixel-level-results .  

### Hyperparameters tuning

The following hyperparameters were tuned one by one and tested on 6 epochs on Cityscapes training before being used for a full training over 50 epochs:
- learning rate: 5e-4
- batch size: 4
- L2 regularization: none
- init of convolutional layers in the decoder part: tf.truncated_normal_initializer(stddev = 0.001)
- dropout: 80%

The settings of initializations and learning rate was very important to achieve good results and fast convergence.

Different optimizers, Adam, Momentum, RMSProp and GradientDescent were tested. Adam provided the best results.

### Training strategy

The loss function used for training is the Cross-entropy evaluated over the training set.  
The IOU is evaluated over the validation set at the end of every epoch, and everytime IOU is improved, the network parameters are saved. So we end up with the best network as per IOU over validation set estimation.  We want to derive a network that is best performing on data not part of the training set.  
When the IOU does not improve at the end of an epoch, the learning rate is divided by a rather big factor of 2.  
Early stop is being used so that the training stops automaticaly when no improvement in terms of IOU over the validation set is reported during 3 consecutive epochs. This enables to prevent overfitting (increasing performance over training set while decreasing performance over data not, part of the training.  

### Topics for further improvements

- Use a weighted loss function: so that bigger classes do not take over classes with fewer samples and pixels 
- Use a weighted IOU metric
- Add images augmentation
- Provide per class accuracy
- Use a subsampling factor of 2 instead of 4
- Use a different front end in the FCN8s architecture: e.g. MobileNet
- Investigate ENet and ERFNet
- Try out 8 bits quantization further: e.g. with the next Tensorflow release (so far it is slower)

### References

Fully Convolutional Networks for Semantic Segmentation:  
https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf  

MultiNet: Real-time Joint Semantic Reasoning for Autonomous Driving:  
https://arxiv.org/pdf/1612.07695.pdf

ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation:  
https://arxiv.org/pdf/1606.02147.pdf  

ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation:  
http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf  

A 2017 Guide to Semantic Segmentation with Deep Learning:  
http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review  
  
CS231n: Convolutional Neural Networks for Visual Recognition  
http://cs231n.stanford.edu/syllabus.html  



### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Datasets
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.  
Download the Cityscapes data set from here: https://www.cityscapes-dataset.com/ . Extract the dataset in the `data/cityscapes` folder. This will create the folders `leftImg8bit` for the train, val and test sets and `gtFine` for the associated ground truth images.  



### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
