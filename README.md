# Position Estimation with CNN
This is an implementation of position estimation inspired by [PoseNet](http://mi.eng.cam.ac.uk/projects/relocalisation/).

## Introduction
 The authors of [PoseNet](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kendall_PoseNet_A_Convolutional_ICCV_2015_paper.pdf) used a large Convolutional Neural Network (CNN), the [GoogLeNet](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf) and trained it in an end-to-end manner with large image dataset in order to predict positions and the orientation.
 
 In the [paper](https://www.mad.tf.fau.de/files/2018/07/Evaluation-Criteria-for-Inside-Out-Indoor-Positioning-Systems-based-on-Machine-Learning.pdf) the authors use the data published by the [Fraunhofer IIS](https://www.iis.fraunhofer.de/en/ff/lv/dataanalytics/tech/opt/warehouse.html) for position estimation in indoor enviornments. 

The authors scale the images from original size of 640x480. 

![Initial image](/pictures/img2.png)

Then the image is center-cropped to size of 224x224 as input for the CNN.

![Center cropped](/pictures/img3.png)

Additionally the datasets mean image is being substracted from input.

![Final image](/pictures/img4.png)

## Data
The data used in this project is provided by [Fraunhofer IIS](https://www.iis.fraunhofer.de/en/ff/lv/dataanalytics/tech/opt/warehouse.html).

### Requirements
```
pip install tqdm
pip install statsmodels
pip install tensorflow-gpu==1.8
pip install keras
pip install opencv-python
```
### Setup
* Clone the repository $cv_position

* Download the datasets for training:
```
cd $cv_position
wget https://www2.iis.fraunhofer.de/IPIN/training/horizontal.tar.gz
wget https://www2.iis.fraunhofer.de/IPIN/training/vertical.tar.gz
```
* For testing we download (can be any of them):
```
wget https://www2.iis.fraunhofer.de/IPIN/testing/cross.tar.gz
```
* Move the tarred files into **data/** folder
```
mv *.tar.gz data/
```
* Unpack the data files
```
tar -xvzf [horizontal.tar.gz, vertical.tar.gz, cross.tar.gz]
```
### Network architecture
The original PoseNet has 12,431,685 trainable parameters, training it takes awhile. I want to show that with much smaller network it is possible to achieve comparable results as reported in the [paper](https://www.mad.tf.fau.de/files/2018/07/Evaluation-Criteria-for-Inside-Out-Indoor-Positioning-Systems-based-on-Machine-Learning.pdf).
**smallNet** consist of fewer convolutional layers has 3,218,279 trainable parameters.

<p align="center">
  <img width="500" height="500" src="https://github.com/kenkyusha/cv_position/blob/master/pictures/model_plot.png?raw=true">
</p>

### GPU
I used NVIDIA GeForce GTX 1060 GPU (6144 MB) with Intel® Core™ i7-7700HQ Quad-Core Processor.
### Training a model
When I investigate the input data presented to the network in the [work](https://www.mad.tf.fau.de/files/2018/07/Evaluation-Criteria-for-Inside-Out-Indoor-Positioning-Systems-based-on-Machine-Learning.pdf), I can only guess that the deep CNN does not really know what to look for. The images are losing partial information (center-cropping) and furthermore substracting the mean image will cause the input looking rather blurry and bad. 
My hypothesis is that the results presented in the [paper](https://www.mad.tf.fau.de/files/2018/07/Evaluation-Criteria-for-Inside-Out-Indoor-Positioning-Systems-based-on-Machine-Learning.pdf) can be improved upon.
Meaning instead of using the center crop and mean reduction, I only scale the image directly down to 224x224 and instead of removing the mean image, use edge detection (cv2.canny). I trained the **smallNet** with **horizontal** and **vertical** datasets using the mentioned image preprocessing.

![Proposed image](/pictures/img5.png)

* Run the script **train_net.py** (uses GPU if available):
```
python train_net.py --model [MODEL] --test_data [TEST-DATA-LIST]  --data [TRAIN-DATA-LIST]
python train_net.py --model smallNet --test_data data/raw_cross_sys4.txt  --data data/full_dataset_train.txt
```
The **full_dataset_train.txt** consist of image file paths to the horiztonal and vertical data sets with their corresponding label. The dataset combined is approximately 200,000 images. For testing the model, we use the cross with sys8 (orientation of the image).

### Testing a model

* Run the script **pred_on_net.py** (uses GPU if available):
```
python pred_on_net.py --data [TEST-DATA-LIST] --net [PATH2MODEL] --wts [PATH2WEIGHTS] --fname [NAME]
python pred_on_net.py --data data/raw_cross_sys4.txt --net example/net_smallNet.h5 --wts example/wts_smallNet.h5
```

### Results
Preliminary results on the cross dataset

Implemenatation | MAE    | CEP    | CE95   | deg (CEP)
--------------- | ------ | -------| ------ | ---------
Cross (original)| 1.08 m | 0.86 m | 3.06 m | 0.18 ° 
smallNet (sys4) | 0.35 m | 0.30 m | 0.74 m | 0.10 °
smallNet (sys5) | 0.37 m | 0.35 m | 0.71 m | 0.04 °
smallNet (sys7) | 0.59 m | 0.43 m | 1.48 m | 0.03 °

![sys4](/pictures/net_smallNet_pred_raw_cross_sys4.png)
![sys5](/pictures/net_smallNet_epoch_pred_raw_cross_sys5.png)
![sys7](/pictures/net_smallNet_pred_raw_cross_sys7.png)

### Analysis
Thinking about this problem of indoor positioning, where we measurements (images) are taken at each position covering almost 360 ° field of view should work very well using data-driven approaches. I can only think of that the CNN trained with this kind of data and target values (labels), is getting little bit confused on what to look in an image. If I picture myself in a room and try to understand the relation of objects and walls around me, then first thing I think about are the key features. Just as if one is looking at a city map of the place where they are travelling to, we look for certain details such as monuments, significant structures etc for understanding where we are located. This itself gave me an idea that perhaps the CNN can be tought by just showing the lines or the edges of the objects in the images. These lines will look more or less the same if the position is moving towards them or away from them meaning translational invariance and this is exactly something which CNN-s are really good at.

TODO: 
- Grab images of the testing set (cross) and their labels (ca 1500) and 
- find labels from the training set (horizontal and vertical), which are close to the testing images
- compare the images to see whether they are similar (CHECK for overfitting)

* GPU metrics
* visualizing CNN kernels 
