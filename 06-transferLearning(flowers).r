# A neural network is trained on a data. This network gains knowledge from this
# data, which is compiled as “weights” of the network. These weights can be 
# extracted and then transferred to any other neural network. Instead of 
# training the other neural network from scratch, we “transfer” the learned 
# features.


# three step process; 
# 1) load an existing model and add some layers, 
# 2) train the extended model on your own data, 
# 3) set more layers trainable and fine-tune the model on your own data.

require(keras)

################### Section 1 #########################

train_directory="C://Users//user//Desktop//DataBases//datasets//deepLearning//KerasDeepLearning//17flowers//imgs//"

# create a folder called flower on your desktop with 
# test and train folder inside

flower_train="C://Users//user//Desktop//flower//train//"
flower_test="C://Users//user//Desktop//flower//test//"

flower_names=readLines("C://Users//user//Desktop//DataBases//datasets//deepLearning//KerasDeepLearning//17flowers//flower_metadata.txt")


for(i in 1:length(flower_names)){
dir.create(paste0(flower_train,flower_names[i], sep=""))
}


for(i in 1:length(flower_names)){
  dir.create(paste0(flower_test,flower_names[i], sep=""))
}



flowernames=list.files(train_directory)
flowerlength=length(list.files(train_directory))

### populate the train directory

i=1
j=1
while(i<flowerlength){
    original_dir=train_directory
    target_dir=paste0(flower_train,flower_names[j],sep="")
    k=i+59
    listfiles=flowernames[i:k]
  files_to_copy=paste0(original_dir,listfiles, sep="")
  file.copy(from=files_to_copy,
            to=target_dir,
            overwrite=FALSE, 
            recursive=FALSE,
            copy.mode=TRUE)
  i=i+80
  j=j+1
}



### populate the test directory

ii=60
jj=1
while(ii<flowerlength){
  original_dir=train_directory
  target_dir=paste0(flower_test,flower_names[jj],sep="")
  kk=ii+19
  listfiles=flowernames[ii:kk]
  files_to_copy=paste0(original_dir,listfiles, sep="")
  file.copy(from=files_to_copy,
            to=target_dir,
            overwrite=FALSE, 
            recursive=FALSE,
            copy.mode=TRUE)
  ii=ii+80
  jj=jj+1
}




img_width=500
img_height=575
batch_size=10


train_generator <- flow_images_from_directory(flower_train, 
                                              generator = image_data_generator(),
                                              target_size = c(img_width, img_height), 
                                              color_mode = "rgb",
                                              batch_size=batch_size,
                                              class_mode = "categorical", 
                                              shuffle = TRUE,
                                              seed = 123)

validation_generator <- flow_images_from_directory(flower_test, generator = image_data_generator(),
                                                   target_size = c(img_width, img_height), 
                                                   color_mode = "rgb",
                                                   class_mode = "categorical", 
                                                   batch_size = batch_size, 
                                                   shuffle = TRUE,
                                                   seed = 123)

train_samples = 1020
validation_samples = 340

################### Section 2 #########################

base_model <- application_vgg16(
  weights = 'imagenet', 
  include_top = FALSE)

################### Section 3 #########################
## replace the softmax layer

predictions <- base_model$output %>% 
  layer_global_average_pooling_2d(trainable = T) %>% 
  layer_dense(256, trainable = T) %>%
  layer_activation("relu", trainable = T) %>%
  layer_dropout(0.5, trainable = T) %>%
  layer_dense(17, trainable=T) %>%    ## important to adapt to fit the 17 classes in the dataset!
  layer_activation("softmax", trainable=T)

# this is the model we will train
model <- keras_model(inputs = base_model$input, outputs = predictions)


################### Section 4 #########################
### freeze the model

for (layer in base_model$layers)
{layer$trainable <- FALSE}


################### Section 5 #########################
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adadelta',
  metrics = "accuracy"
)


#################### Section 6 #######################

#================================================================
#             THIS WILL TAKE ABOUT 8 hours
#===============================================================

model %>% fit_generator(
  train_generator,
  steps_per_epoch = as.integer(train_samples/batch_size), 
  epochs = 4, 
  validation_data = validation_generator,
  validation_steps = as.integer(validation_samples/batch_size),
  verbose=1
)

################### FINE TUNING ######################

for (layer in base_model$layers[1:5])
{layer$trainable <- FALSE}

for (layer in base_model$layers[5:length(base_model$layers)])
{layer$trainable <- TRUE}



################### Section 5 #########################
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_sgd(lr = 0.0001, decay = 1e-6),  ## play with the learning rate
  metrics = "accuracy"
)

############ saving and loading model

# save_model_hdf5(model, "my_model_flowers.h5")
# model <- load_model_hdf5("my_model_flowers.h5")

#================================================================
#             THIS WILL TAKE ABOUT 2 hours
#===============================================================


model %>% fit_generator(
  train_generator,
  steps_per_epoch = as.integer(train_samples/batch_size), 
  epochs = 1, 
  validation_data = validation_generator,
  validation_steps = as.integer(validation_samples/batch_size),
  verbose=1
)


############# predictions

img <- image_load(file.choose(), target_size = c(224,224))
x <- image_to_array(img)
dim(x) <- c(1, dim(x))
prediction <- model %>% predict(x)

colnames(prediction) <- list.files(flower_train)
prediction[,which.max(prediction)]
