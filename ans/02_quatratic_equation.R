X = seq(from=0, to=10, length.out=200)
MyFunction <- function(x, a=-1, b=0, c=1){
  a + b*x + c*x^2 
}
Y=MyFunction(X)


x_train=X
y_train=Y


######### step2 > build the NN model

model <- keras_model_sequential()

model %>%
  layer_dense(
    units=100,
    activation='relu',
    input_shape=c(1)) %>%
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
  epochs=50,
  verbose = 1,
  validation_split=0.2     #validation_data = list(x_test, y_test)
)

# verbose: 0, 1, or 2. 
# Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch

plot(mymodel)

score <- model %>% evaluate(
  x_train, y_train)

print(score)


############ saving and loading model

# save_model_hdf5(model, "my_model.h5")
# model <- load_model_hdf5("my_model.h5")

############# predictions

pred=model %>% predict(x_train)
head(pred)

plot(pred,y_train)

plot(1:length(y_train), y_train)
points(1:length(pred), pred, col="red")
