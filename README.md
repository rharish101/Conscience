# Conscience

This is my code for question 1 of the Machine Learning event called Conscience held in Takneek '17 at IIT Kanpur.

Contents
--------

* *q1.py*: The main file to train a fine-tuned VGG16 model on the given dataset.
* *256_ObjectClassification.zip*: The dataset for question 1, a set of images classified into 257 categories (not 256). Download from *https://drive.google.com/drive/folders/0B8ZTgwx4fdLua1RIM3RxNG5BOUk* (if link is broken, then email me).
* *q1_vgg_mod.gz0x*: Split parts of a tarball containing the final fine-tuned VGG16 model, a Keras Sequential model. Concatenate the split parts into one .tar.gz file and then extract it.
* *q1_dense_top.h5*: The Keras Sequential model for the fully-connected layers to be used instead of VGG16's fully-connected layers.
* *q1_dense_top_weights.h5*: The Keras Sequential weights for the fully-connected layers to be used instead of VGG16's fully-connected layers.
* All other files are numpy arrays stored for ease of access when debugging or running the code again.
