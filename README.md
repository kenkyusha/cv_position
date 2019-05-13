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

### Training a model
My hypothesis is that the results presented in the [paper](https://www.mad.tf.fau.de/files/2018/07/Evaluation-Criteria-for-Inside-Out-Indoor-Positioning-Systems-based-on-Machine-Learning.pdf) can be improved upon preserving the image shape. Meaning instead of using the center crop and mean reduction, I only scale the image directly down to 224x224 and instead of removing the mean image, use edge detection (cv2.canny). I trained the **smallNet** with **horizontal** and **vertical** datasets using the mentioned image preprocessing.

![Proposed image](/pictures/img5.png)

* Run the script **train_net.py** (uses GPU if available)
```
python3 train_net.py --model smallNet --test_data data/raw_cross_sys8.txt  --data data/full_dataset_train.txt
```
### Testing a model

* Run the script **pred_on_net.py** (uses GPU if available)
```
python3 pred_on_net.py --data data/raw_cross_sys3.txt --net wts/net_smallNet_epoch_224.h5 --wts wts/wts_smallNet_epoch_224.h5 --fname smallCross_sys3
```

### Analysis
TODO: 
- Grab images of the testing set (cross) and their labels (ca 1500) and 
- find labels from the training set (horizontal and vertical), which are close to the testing images
- compare the images to see whether they are similar (CHECK for overfitting)
