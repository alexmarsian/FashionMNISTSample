# FashionMNISTSimple

This repo has been shared with permission of the Course convenor for COMP5318 (Machine learning and data mining) at the University of Sydney. Reproducing code here for other assignments/courses at the University of Sydney will be considered plagiarism.

This work was completed as my individual contribution to a group project. 

Constraints:

For this assignment, we were required to build deep learning models to predict all 100 classes of the Cifar100 dataset: https://www.cs.toronto.edu/~kriz/cifar.html 

We were encouraged to replicate model architectures from papers, and to try and create a novel architecture. I created an architecture called ParaNet which aims to learn at different spatial scales simultaneously, before combining. It achieves similar results to a standard ResNet44 (the 44-layer was described in the original ResNet paper for use with the Cifar dataset). ParaNet is designed this way to try and handle the smaller 32 x 32 images better than architectures that have been designed for the imageNet competition. The ParaNet architecture needs heavy tweaking, as I only played with the overall depth and not other factors such as drop out, normalisation, feed back loops, or residual connections. 

This project also includes different learning rate schedules implemented in Keras, with a Keras implementation of Leslie Smith's OneCycle policy: https://arxiv.org/pdf/1803.09820.pdf, which was popularised by FastAI. There is an accompanying written report which contains all references (most should be included in the ipython notebooks however) that I can share upon request. 

Languages/Libraries:

All work was done in Python3 with deep learning architectures built using TensorFlow and Keras. 

