# cmd line >> (python/conda) install tensorboard


#The Keras R interface uses the TensorFlow backend engine by default. 
#To install both the core Keras library as well as the TensorFlow backend use the install_keras() function:

install.packages(c("keras", "tfruns","tfestimators"), dep=T)


# install keras package
library(keras)
install_keras()

## install keras package (GPU support)
#install_keras(tensorflow="gpu")


# Here are some examples where you load in the MNIST, CIFAR10 and IMDB data with the keras package:

# Read in MNIST data              # http://yann.lecun.com/exdb/mnist/
mnist <- dataset_mnist()

# Read in CIFAR10 data            # https://www.cs.toronto.edu/~kriz/cifar.html
cifar10 <- dataset_cifar10()

# Read in IMDB data
imdb <- dataset_imdb()



######### troubleshoot

library(reticulate)
py_discover_config('keras')
py_discover_config('tensorboard')

is_keras_available() 
