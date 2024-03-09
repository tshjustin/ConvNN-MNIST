## Testing with CNN

Testing convolutional neural network model for the identification of handwritten digits. 

### Model used 
A CNN model of 2 blocks of Conv-Maxpool pair are used, with the layers having 12 kernels each. An optimizer of SGD & loss function of Cross Entropy loss is used.

### Implementation 
The model is implemented twice, once in Pytorch and the other in Tensorflow. Both accuracy are around the same at 98%. 

More kernels are added to the layer but with no apparent increase in accuracy, but adding another convolutional layer improved the accuracy to 99%. 

**Demo on Render**

https://convnn-mnist.onrender.com/
