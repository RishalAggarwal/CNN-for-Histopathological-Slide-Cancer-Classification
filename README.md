# CNN-for-Histopathological-Slide-Cancer-Classification

Diagnosis of the type of breast cancer using histopathological slides and Deep CNN features

## Dataset

The dataset is described in the following paper:

*Spanhol, Fabio & Soares de Oliveira, Luiz & Petitjean, Caroline & Heutte,
Laurent. (2015). A Dataset for Breast Cancer Histopathological Image
Classification.*

This dataset contains 7909 breast cancer histopathology images acquired
from 82 patients. The dataset contains both malignant and benign images.

## Data Preprocessing

### Normalisation

In order to quantitatively analyse histopathology slides using computer
vision methods, color and stain intensity consistencies need
to be achieved. Inconsistencies mainly occur due to variations while
preparing slides as well as storage time and conditions. Therefore these
intensities need to be normalised and the associated stain images need to
be isolated for further processing.

Normalisation and extraction of stain slides is done via Macenko normalisation. The method of the following is give in *M. Macenko, M. Neithammer, J.S. Marron, D. Borland, J. T. Woosley, X.
Guan, C. Schmitt, and N. E. Thomas. A method for normalizing histology
slides for quantitative analysis.* 

The following examples show the effect of this method
wherein the original image is plotted alongside the hematoxylin stain (affinity for nucleic acids) image that
has been extracted from it.

![alt text](https://github.com/RishalAggarwal/CNN-for-Histopathological-Slide-Cancer-Classification/blob/master/norm_slide.png)

### Augmentation

A 60-20-20 train,test,validation split was used. Furthermore multpile augmentation techniques such as random rotations, hieght and width shifts,horizontal and vertical flips were used.

## Classification

VGG-7 architecture, Resnet and Resnet-inception V4 classifiers were used for the classification task. They all showed results near 82% accuracy. To check the areas of interest identified by the VGG classifier Class Activation Maps (CAMs) were built. These CAMs showed that the classifier does not have a well defined area of interest and thus these areas will need to be localised before applying the classification task.An example of a Class Activation Map that was extracted is given below.

![alt text](https://github.com/RishalAggarwal/CNN-for-Histopathological-Slide-Cancer-Classification/blob/master/CAM%20imgs/40x/benign/adenosis/SOB_B_A-14-29960AB-40-008_cam.png)

## Future Steps

On consulting a pathologist it was decided that areas with large concentrations of nucleii could be ares of interests for this task. Therefore an algorithm has to be designed to localise these areas possibly through a clustering algorithm after approximate positions of nucleii are found using a LOG filter or other methods. Examples of these areas of interests are given in the image below.

![alt text](https://github.com/RishalAggarwal/CNN-for-Histopathological-Slide-Cancer-Classification/blob/master/ROI_slide.PNG)

This image has been borrowed from the paper *Breast cancer multi-classification from histopathological images with
structured deep learning model. Z Han, B Wei, Y Zheng, Y Yin, K Li, S Li*
