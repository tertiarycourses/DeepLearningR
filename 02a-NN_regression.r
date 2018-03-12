library(keras)

######### Step1 > preprocess the data

#preprocess the training_data

train=read.csv(file.choose())                        # sales__data_testing
train=as.matrix(train)
dimnames(train)=NULL
#To use the normalize() function from the keras package, 
# you first need to make sure that you're working with a matrix.
train_sc=normalize(train[,c(1:9)])   # scale to 0min and 1max

#preprocess the testing data
test=read.csv(file.choose())                        # sales_data_training
test=as.matrix(test)
dimnames(test)=NULL

test_sc=normalize(test[,c(1:9)])

## create input and output for model testing
x_train=train_sc
y_train=train[,10]


x_test=test_sc
y_test=test[ ,10]

######### step2 > build the NN model

model <- keras_model_sequential()

model %>%
    layer_dense(
          units=50,
          activation='relu',
          input_shape=c(9)) %>%
    layer_dense(
           units=100,
           activation='relu') %>%
     layer_dense(
           units=200,
           activation='relu') %>%
     layer_dense(
           units=1,
           activation='linear')

# linear for regression /softmax for classification

summary(model)

############# compile (define loss and optimizer)

model %>% compile(
       loss='mean_squared_error', 
       optimizer='adam',                               
       metrics=c('mse')
)

########### Train and evaluate

mymodel= model %>% fit(
     x_train, y_train,
     batch_size=10,          
      epochs=50,
     verbose = 1,
     validation_split=0.2     #validation_data = list(x_test, y_test)
)

# verbose: 0, 1, or 2. 
# Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch

plot(mymodel)

error <- model %>% evaluate(
       x_test, y_test,
       batch_size=9)

print(error)


############ saving and loading model

# save_model_hdf5(model, "my_model.h5")
# model <- load_model_hdf5("my_model.h5")

############# predictions

pred=model %>% predict(x_test)
head(pred)

plot(pred,y_test)

plot(1:length(y_test), y_test)
points(1:length(pred), pred, col="red")


#=================================================
#            CHALLENGE
#=================================================

# Create a one hidden layer NN to model the following data

X = seq(from=0, to=10, length.out=200)
MyFunction <- function(x, a=-1, b=0, c=1){
  a + b*x + c*x^2 
}
Y=MyFunction(X)

plot(X,Y)         