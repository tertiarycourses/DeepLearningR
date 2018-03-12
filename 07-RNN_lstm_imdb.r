### Trains a LSTM on the IMDB sentiment classification task.

library(keras)

max_features <- 20000
batch_size <- 32

# Cut texts after this number of words (among top max_features most common words)
maxlen <- 80  


# Read in IMDB data
imdb <- dataset_imdb(num_words = max_features)
x_train <- imdb$train$x
y_train <- imdb$train$y
x_test <- imdb$test$x
y_test <- imdb$test$y

### prepare the dataset
x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)


### Define the model
model <- keras_model_sequential()


model %>%
  layer_embedding(input_dim = max_features, output_dim = 128) %>% 
  layer_lstm(units = 64, dropout = 0.2, recurrent_dropout = 0.2) %>% 
  layer_dense(units = 1, activation = 'sigmoid')

### compile the model

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)



############ saving and loading model

# save_model_hdf5(model, "my_model.h5")
# model <- load_model_hdf5("my_model.h5")

############################ train

#==================================================
#   WARNING > this will take 30 mins
#===================================================


mymodel=model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = 15,
  validation_split=0.2        #validation_data = list(x_test, y_test)
)

plot(mymodel)


######################## evaluate

score <- model %>% evaluate(
  x_test, y_test,
  batch_size = batch_size
)

print(score)

######################## predictions

pred=model %>% predict_classes(x_test)
head(pred)

table(pred, imdb$test$y)
mean(pred== imdb$test$y)
