# Finding Pulsars

## Abstract
In this project, we are going to explore the pattern of pulsars which differentiates them from the non-pulsars in a large dataset. Statistical methods are used to illustrate the overall features of each data sample, followed by typical machine learning techniques which fit the pattern with powerful models, and the deep learning method as a convenient tool to deal with complicated features. This model achieves satisfying performance in the test set.

![](/report/1.PNG)
![](/report/2.PNG)
![](/report/3.PNG)
![](/report/4.PNG)
![](/report/5.PNG)
![](/report/6.PNG)

## Problem Analysis
There are three different-source pulsar datasets: 
HTRU1, p309, and PMPS. Since HTRU1 is highly imbalanced (which has far more negative samples then the positive ones), this experiment only explore the data from p309 and PMPS (both of which are in "PFD" data format). 
p309p contains 2698 pulsar samples and 1655 non-pulsar samples.
PMPS contains 1001 pulsar samples and 1001 non-pulsar samples.


## Feature Extraction
Directly using the basic features above to build a model is a little bit impossible, because there are too few features for a machine learning problem and some 2-D or sequence-like features such as the sub-bands and DM curves are difficult to handle as ordinary features. How to extract more features based on the basic infomation is a tough task. And this can be done by creating statistical indicators.





