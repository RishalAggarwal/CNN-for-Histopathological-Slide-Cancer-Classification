# CNN-for-Histopathological-Slide-Cancer-Classification

Diagnosis of the type of breast cancer using histopathological slides and Deep CNN features

## Dataset

The dataset is described in the following paper:

Spanhol, Fabio & Soares de Oliveira, Luiz & Petitjean, Caroline & Heutte,
Laurent. (2015). A Dataset for Breast Cancer Histopathological Image
Classification.

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

Normalisation and extraction of stain slides is done via Macenko normalisation. The method of the following is give in M. Macenko, M. Neithammer, J.S. Marron, D. Borland, J. T. Woosley, X.
Guan, C. Schmitt, and N. E. Thomas. A method for normalizing histology
slides for quantitative analysis. 

The following examples show the effect of this method
wherein the original image is plotted alongside the hematoxylin stain (affinity for nucleic acids) image that
has been extracted from it.

![alt text](https://github.com/RishalAggarwal/CNN-for-Histopathological-Slide-Cancer-Classification/blob/master/norm_slide.png)

