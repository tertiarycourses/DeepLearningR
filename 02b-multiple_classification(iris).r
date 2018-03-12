library(keras)

# Read in `iris` data
data(iris)
iris2=iris
View(iris2)

######## study the data

# Return the first part of `iris`
head(iris2)
# Inspect the structure
str(iris2)
# Obtain the dimensions
dim(iris2)

plot(iris2$Petal.Length, 
     iris2$Petal.Width, 
     pch=21, 
     bg=c("red","green3","blue")[unclass(iris2$Species)], 
     xlab="Petal Length", 
      ylab="Petal Width")

# Overall correlation between `Petal.Length` and `Petal.Width` 
cor(iris2$Petal.Length, iris2$Petal.Width)
# Store the overall correlation in `M`
M <- cor(iris2[,1:4])

# Plot the correlation plot with `M`
library(corrplot)
corrplot(M, method="circle")

########## preprocess the data
iris2[,5] <- as.numeric(iris2[,5]) -1

# Turn `iris` into a matrix
iris2 <- as.matrix(iris2)

# Set iris `dimnames` to `NULL`
dimnames(iris2) <- NULL

# Normalize the `iris` data
iris22 <- normalize(iris2[,1:4])      
#normalize is from the keras package

# Return the summary of `iris`
summary(iris22)

# Determine sample size
ind <- sample(2, nrow(iris22), replace=TRUE, prob=c(0.67, 0.33))

# Split the `iris` data
iris.training <- iris22[ind==1, 1:4]
iris.test <- iris22[ind==2, 1:4]

# Split the class attribute
iris.trainingtarget <- iris2[ind==1, 5]
iris.testtarget <- iris2[ind==2, 5]

# to model multi-class classification problems with neural networks, 
#it is generally a good practice to 
# make sure that you transform your target attribute from a vector 
# that contains values for each class 
# value to a matrix with a boolean for each class value 

# One hot encode training target values
iris.trainLabels <- to_categorical(iris.trainingtarget)         
# to_categorical is from keras package

# One hot encode test target values
iris.testLabels <- to_categorical(iris.testtarget)

# Print out the iris.testLabels to double check the result
print(iris.testLabels)


########### Construct the model

### Initialize a sequential model
model <- keras_model_sequential() 

# Add layers to the model
model %>% 
layer_dense(units = ncol(iris22)*2, activation = 'relu', input_shape = c(ncol(iris22))) %>% 
layer_dense(units = 3, activation = 'softmax')
# because we have 3 species of flowers


### inspect the model

# Print a summary of a model
summary(model)

# Get model configuration
get_config(model)

# Get layer configuration
get_layer(model, index = 1)

# List the model's layers
model$layers

# List the input tensors
model$inputs

# List the output tensors
model$outputs


#######################  Compile And Fit The Model

# Compile the model
model %>% 
compile(
loss = 'categorical_crossentropy',
        optimizer = 'adam',
        metrics = 'accuracy'
)

# Fit the model
mymodel <-model %>% fit(
iris.training, 
iris.trainLabels, 
epochs = 200, 
batch_size = 5, 
validation_split = 0.2
 )


################ Visualilze model training history
#  visualize the fitting 
history=mymodel

plot(history) 

### Plot the model loss of the training data

plot(history$metrics$loss, 
     main="Model Loss", 
      xlab = "epoch", 
      ylab="loss", 
      col="blue", 
       type="l")

# Plot the model loss of the test data
lines(history$metrics$val_loss, col="green")
legend("topright", c("train","test"), col=c("blue", "green"), lty=c(1,1))

### Plot the accuracy of the training data 
plot(history$metrics$acc, 
     main="Model Accuracy", 
     xlab = "epoch", 
     ylab="accuracy", 
     col="blue", 
     type="l")



# Plot the accuracy of the validation data
lines(history$metrics$val_acc, col="green")
legend("bottomright", c("train","test"), col=c("blue", "green"), 
       lty=c(1,1))


###################### Predicting new data

# Predict the classes for the test data
classes <- model %>% 
  predict_classes(iris.test, batch_size = 5)

# Confusion matrix

table(iris.testtarget, classes)
mean(iris.testtarget == classes)


###################### Evaluation

# Evaluate on test data and labels

score <- model %>% evaluate(iris.test, iris.testLabels, 
                            batch_size = 5)

# Print the score
print(score)


############### Challenge #############

# perform a multi-layer perceptron (as above)
# on the diamonds dataset in ggplot2 packages, 
# choose the color column as your
# predictor variable, choose just the first 500 rows for your
# training set (do not use the entire dataset)