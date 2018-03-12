library(keras)
mnist <- dataset_mnist()

# to classify grayscale images of handwritten digits (28 pixels by 28 pixels) 
#into their 10 categories (0 to 9). 

#### prpare the dataset

#separating train and test file
x_train<-mnist$train$x
y_train<-mnist$train$y
x_test<-mnist$test$x
y_test<-mnist$test$y

## look at our data
str(y_train)

str(y_test)

x_train <- array_reshape(x_train, c(nrow(x_train), 784))   # 28 * 28 gives 784
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255


#The y data is an integer vector with values ranging from 0 to 9. To prepare this data for training we 
#one-hot encode the vectors into binary class matrices using the Keras to_categorical() function:
#converting the target variable to once hot encoded vectors using keras inbuilt function


y_train<-to_categorical(y_train,10)
y_test<-to_categorical(y_test,10)


#### building model

model <- keras_model_sequential() %>% 

               layer_dense(units = 1024, 
               activation = "relu", 
                input_shape = c(28 * 28)) %>%

                layer_dense(units = 512, 
               activation = "relu") %>%

                layer_dense(units = 256, 
               activation = "relu") %>%

                layer_dense(units = 128, 
               activation = "relu") %>%

               layer_dense(units = 64, 
               activation = "relu") %>%

               layer_dense(units = 32, 
               activation = "relu") %>% 

               layer_dense(units = 10, 
               activation = "softmax")


##### compile model

# A loss function—How the network will be able to measure how good a job it’s 
#doing on its training data, and thus how it will be able to steer itself in the right direction.
# An optimizer—The mechanism through which the network will update itself 
# based on the data it sees and its loss function.
# Metrics to monitor during training and testing—Here we’ll only care about accuracy 
# (the fraction of the images that were correctly classified).

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)



##### fit our model

mymodel= model %>% fit(
      x_train, y_train,
      batch_size=128,
      epochs=5,
      verbose = 1)



############ saving and loading model

# save_model_hdf5(model, "my_model.h5")
# model <- load_model_hdf5("my_model.h5")


#### evalaute our model

score <- model %>% evaluate(
                    x_test, y_test,
                    batch_size=128)
score


############# predictions

pred=model %>% predict_classes(x_test)
head(pred)

table(pred, mnist$test$y)
mean(pred==mnist$test$y)


#==========================================
#                 CHALLENGE
#==========================================

#Create a 6 layers NN model for MNIST classification and #compare with 4 layers model
#L1 = 1024
#L2 = 512
#L3 = 256
#L4 = 128
#L5 = 64
#L6 = 32


