library(keras)
library(jpeg)


modelresnet50 <- application_resnet50(weights='imagenet')

modelvgg19 <- application_vgg19(weights = 'imagenet')


jj<-readJPEG(file.choose())   # read snake or elephant
plot(0:1,0:1,type='n',ann=FALSE,axes=FALSE)
rasterImage(jj,0,0,1,1)


img <- image_load(file.choose(), target_size = c(224,224))   # choose snake or elephant
x <- image_to_array(img)

dim(x)=c(1, dim(x))
x <- imagenet_preprocess_input(x)


pres_resnet <- modelresnet50 %>% predict(x)
pres_vgg19 <- modelvgg19 %>% predict(x)


ppres=imagenet_decode_predictions(pres_resnet, top=10)[[1]]
ppvgg=imagenet_decode_predictions(pres_vgg19, top=10)[[1]]

ppres
ppvgg


########

# repeat with different cat and dog pics

## very poor in recognizing brand logos > will will improve this with transfer learning later


##### Challenge 
# download your own photo > resize to 224 by 224 pixels
# and try it in the image classifier