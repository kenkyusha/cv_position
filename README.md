# Position Estimation with CNN
This is an implementation of position estimation inspired by [PoseNet](http://mi.eng.cam.ac.uk/projects/relocalisation/).

## Motivation
 The authors of [PoseNet](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kendall_PoseNet_A_Convolutional_ICCV_2015_paper.pdf) used a large Convolutional Neural Network (CNN), the [GoogLeNet](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf) and trained it in an end-to-end manner with large image dataset in order to predict positions and the orientation.
 
 In the [paper](https://www.mad.tf.fau.de/files/2018/07/Evaluation-Criteria-for-Inside-Out-Indoor-Positioning-Systems-based-on-Machine-Learning.pdf) the authors use the data published by the [Fraunhofer IIS](https://www.iis.fraunhofer.de/en/ff/lv/dataanalytics/tech/opt/warehouse.html) for position estimation in indoor enviornments. 

The authors scale the images from original size of 640x480 and use center crop of size 224x224 as input for the CNN. Additionally the datasets mean image is being substracted from input.

## Data
The data used in this project is provided by [Fraunhofer IIS](https://www.iis.fraunhofer.de/en/ff/lv/dataanalytics/tech/opt/warehouse.html).
