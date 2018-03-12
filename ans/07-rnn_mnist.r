### Trains a rnn on the MNIST dataset.

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
x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols))
input_shape <- c(img_rows, img_cols)

# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255

# Convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)


# Define Model -----------------------------------------------------------

model <- keras_model_sequential()

model %>%
  layer_lstm(input_shape=input_shape, 
             units = 64, dropout = 0.2, 
             recurrent_dropout = 0.2) %>% 
  layer_dense(units = 10, activation = 'tanh')%>%
  layer_dense(units = num_classes, 
              activation = 'softmax')



#layer_simple_rnn

#layer_cudnn_gru


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
mean(pred== mnist$test$y)

