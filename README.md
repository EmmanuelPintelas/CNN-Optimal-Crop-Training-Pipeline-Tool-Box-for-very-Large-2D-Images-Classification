# CNN-Training-Pipeline-Tool-Box-for-very-Large-2D-Images-Classification

Training a CNN model with very large images (e.g. 100k x 100k x 4) is impossible even in supercomputers.
The naive approach is just to resize the large image into a viable size like 1000x1000 but then most of the important information will have lost.
Thus, in this project, we mainly provide a _best-crop_ training pipeline for CNN in large image  classification.
More specifically, based on our hand-crafted code we detect the most informative sub-region of the large image and use this for training.
The size of the crop/tile is viable for training a cnn model, e.g. 1000x1000, while the important information have NOT lost for feeding into the CNN.


In this project, we also provide 2 training-optimization pipelines for CNN models
emphasizing on very large images, but it can be used as a general tool.

The first one is a traditional training pipeline  _Train Loop_, which iteratively
trains and save the best validated model.

The second one is  _Train Loop Augments_, which automatically activates random train augmentations
only when the training accuracy becomes too high, in order to add regularlization effect during training to avoid overfitting 
and also boost even more the validation performance.


