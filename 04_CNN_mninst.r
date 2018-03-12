browseURL("http://yann.lecun.com/")                        # homepage
browseURL("http://yann.lecun.com/exdb/lenet/index.html")   # demo

### Trains a simple convnet on the MNIST dataset.

library(keras)

# Data Preparation -----------------------------------------------------

# The data, shuffled and split between train and test sets
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

batch_size <- 100
num_classes <- 10
epochs <- 2    # small number coz training takes a long time

# Input image dimensions
img_rows <- 28
img_cols <- 28

# Redefine  dimension of train/test inputs
x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)

# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255


# Convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)

# Define Model -----------------------------------------------------------

model <- keras_model_sequential()


model %>%
  layer_conv_2d(filters = 32,                    # filters not neurons
                kernel_size = c(3,3),
                padding = "same",
                activation = 'relu',
                input_shape = input_shape) %>% 
   layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
   layer_conv_2d(filters = 64,                    # filters not neurons
                kernel_size = c(3,3),
                padding = "same",
                activation = 'relu') %>% 
    layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
    layer_conv_2d(filters = 128,                    # filters not neurons
                kernel_size = c(3,3),
                padding = "same",
                activation = 'relu') %>% 
     layer_dropout(rate = 0.5) %>%
    layer_flatten() %>% 
    layer_dense(units = 256,                        # NN layer
              activation = 'relu') %>% 
    layer_dense(units = num_classes, 
              activation = 'softmax')


summary(model)

# Compile model -----------------------------------------------
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics ='accuracy'
)

# Train & Evaluate -------------------------------------------------------
### !!!! this will take about 5 minutes !!!

mymodel = model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 1,
  validation_split=0.2   #validation_data = list(x_test, y_test)
)

plot(mymodel)


############ saving and loading model

# save_model_hdf5(model, "my_model.h5")
# model <- load_model_hdf5("my_model.h5")


#Evaluating model on the cross validation dataset
score <- model %>% evaluate(
  x_test, y_test)

score

############# predictions

pred=model %>% predict_classes(x_test)
head(pred)

table(pred, mnist$test$y)
mean(pred==mnist$test$y)

# =======================================================================
#    CHALLENGE
# =======================================================================

# Create a CNN for the CIFAR-10 dataset on Keras:
# • conv layer - 32 filters,size=3,relu
# • max pooling layer - size=2
# • conv layer - 64 filters,size=3,relu
# • max pooling layer - size=2
# • conv layer - 128 filters,size=3,relu
# dropout 0.5
# flatten
# • fc layer - 256 neurons, relu
# • fc softmax output layer