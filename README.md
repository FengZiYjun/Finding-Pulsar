# Finding Pulsars

## Abstract
In this project, we are going to explore the pattern of pulsars which differentiates them from the non-pulsars in a large dataset. Statistical methods are used to illustrate the overall features of each data sample, followed by typical machine learning techniques which fit the pattern with powerful models, and the deep learning method as a convenient tool to deal with complicated features. This model achieves satisfying performance in the test set.

## Problem Analysis
There are three different-source pulsar datasets: 
HTRU1, p309, and PMPS. Since HTRU1 is highly imbalanced (which has far more negative samples then the positive ones), this experiment only explore the data from p309 and PMPS (both of which are in "PFD" data format). 
p309p contains 2698 pulsar samples and 1655 non-pulsar samples.
PMPS contains 1001 pulsar samples and 1001 non-pulsar samples.

![](/report/1.PNG)
![](/report/2.PNG)
![](/report/3.PNG)
![](/report/4.PNG)
![](/report/5.PNG)
![](/report/6.PNG)

## Feature Extraction
Directly using the basic features above to build a model is a little bit impossible, because there are too few features for a machine learning problem and some 2-D or sequence-like features such as the sub-bands and DM curves are difficult to handle as ordinary features. How to extract more features based on the basic infomation is a tough task. And this can be done by creating statistical indicators.

![](/report/7.PNG)
![](/report/8.PNG)


## Classification & Evaluation

### Bagging 
![](/report/9.PNG)
![](/report/10.PNG)

### Boosting
Boosting is also a common ensemble model. In this project, the gradient boosting decision trees and AdaBoost are used. They both have a good performance.

For gradient boosting decision trees, logistic regression for classification with probabilistic outputs acts as a loss function. Gradient boosting is fairly robust to over-fitting so the model use 200 estimators for boosting with learning rate 0.1.
The model is trained and tested in p309 and PMPS.
![](/report/12.PNG)

For Adaboost, the maximum number of estimators is set to be 200 with SAMME algorithm.
The base estimator is a 3-depth decison tree. The model is trained and tested in p309 and PMPS.
![](/report/13.PNG)

![](/report/10.PNG)

### Convolutional Neural Network
The last method is the deep learning approach to deal with the sub-ints and sub-bands images. By linear interpolation along rows, sub-ints can be presented as a 64 × 64 matrix and sub-bands as a 32 × 64 matrix. Hence, the architechture of the Convolutional neural network is as follows

![](/report/14.PNG)

The input is the 32 × 64 image pixels and the output is a scalar - 0 or 1. The network is implemented using TensorFlow with batch normalization and drop-out technique in order to prevent overfitting. The hyper-paramters are determined by experience and instinct. Maybe there is a better parameter combination that has not been tried in this experiment. By adding the convolutonal neural network into the codes before, the model is a combination of a CNN and a random forest, with sub-bands feature only used in CNN. The model achieves satisfying performance. The combined model is trained and tested in p309 and PMPS.

![](/report/15.PNG)