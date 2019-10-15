library(readr)
library(caret)
library(ggplot2)
library(mlbench)
library(e1071)

#Load data----
existingproductattributes2017 <- read_csv("existingproductattributes2017.csv")
Exist <- existingproductattributes2017

#Preprocessing----
# dummify the data
DummyVarsExist <- dummyVars(" ~ .", data = Exist)
readyData <- data.frame(predict(DummyVarsExist, newdata = Exist))

# Final selection relevant features----
Final_relevant_vars <- c(
  "ProductTypeLaptop","ProductTypeNetbook","ProductTypePC",
  "ProductTypeSmartphone","ProductNum","x1StarReviews","x4StarReviews",
  "x3StarReviews","PositiveServiceReview","Volume"
)

# create correlation matrix----
corrData <- cor(readyData[Final_relevant_vars])
final_df <- readyData[Final_relevant_vars]

head(final_df)


set.seed(15)

#create a 20% sample of the data----
BWsample <- final_df[sample(1:nrow(final_df), 70,replace=FALSE),]

# define an 75%/25% train/test split of the dataset----
inTraining_lm <- createDataPartition(BWsample$Volume, p = .75, list = FALSE)
training_lm <- BWsample[inTraining,]
testing_lm <- BWsample[-inTraining,]

#CV 10 fold
fitControl_lm <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

#### MODELLING ####

#LINEAR MODEL----
#lm model: lmfFit AUTOM GRID
#type: line y based on x model
#package: baseR 
#dataframe = final_df
#Y Value = Volume

lmFit <- lm(Volume~., 
   data = training_lm)

#training results
lmFit
saveRDS(lmFit, file = "lmFit.rds")
#LM summary lmFit----
summary(lmFit)
#summaryperformance_lmFit
#multiple R-squared   Adjusted R-squared 
# 0.8699                0.8433 
saveRDS(object = lmFit, file = "lmFit.rds")

#Predict Output----
predicted= predict(lmFit, testing_lm)
print(predicted)
str(predicted)
#not sure why this is helpful since the outcome is hard to 'read'

#LM postresample----
postResample(pred = predict(object = lmFit, newdata = testing_lm), obs = testing_lm$Volume)
##output = RMSE   Rsquared    MAE
##lmFit =  434.867  0.5694    273.809


#KNN MODEL----
#K-nn model: KNNfFit AUTOM GRID
#type: neighbour based model
#package: caret
#dataframe = final_df
#Y Value = Volume

set.seed(15)

#SET SPLIT 75%/25% for train/test in the dataset
inTrainingKNN <- createDataPartition(BWsample$Volume, p = .75, list = FALSE)
trainingKNN <- BWsample[inTraining,]
testingKNN <- BWsample[-inTraining,]

#10 fold cross validation
fitControlKNN <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

#train knn model with a tuneLenght = `1`(trains with 1 mtry values for knn)
#  preProcess=c("center", "scale") removed because not appl on prod types
KNNFit <- train(Volume~., 
                data = trainingKNN, 
                method = "kknn", 
                trControl=fitControlKNN, 
                tuneLength = 1
                )

#training results
KNNFit
#KNN traning results----
#   RMSE     Rsquared  MAE 
#   1215.96  0.8769   MAE 617.10

#KNN summary KNNFit K3----
summary(KNNFit)
#summaryperformance_KNNFit= Min Mean Abs Error: 468.16, Min Mean S-error 1914 
saveRDS(object = KNNFit, file = "KNNFit.rds")

#KNN postresample----
postResample(pred = predict(object = KNNFit, newdata = testingKNN), obs = testingKNN$Volume)
#   RMSE     Rsquared MAE 
#   406.473  0.6072   206.83 

#Predict Output----
predicted= predict(KNNFit, testingKNN)
print(predicted)
str(predicted)
#not sure why this is helpful since the outcome is hard to 'read'



#RF MODEL----
#Random Forest model: rfFit AUTOM GRID
#type: decision tree for mean prediction of individual trees
#package: caret
#dataframe = final_df
#Y Value = Volume

set.seed(15)

#SET SPLIT 75%/25% for train/test in the dataset
inTrainingrf <- createDataPartition(BWsample$Volume, p = .75, list = FALSE)
trainingrf <- BWsample[inTraining,]
testingrf <- BWsample[-inTraining,]

#10 fold cross validation
fitControlrf <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

#train knn model with a tuneLenght = `1`(trains with 1 mtry values for knn)
#  preProcess=c("center", "scale") removed because not appl on prod types
rfFit <- train(Volume~., 
                data = trainingrf, 
                method = "rf", 
                trControl=fitControlrf, 
                tuneLength = 1
)

#training results
rfFit
#RF traning results----
#   RMSE     Rsquared  MAE 
#   881.01   0.9837   447.11
saveRDS(object = rfFit, file = "rfFit.rds")

#RF postresample----
postResample(pred = predict(object = rfFit, newdata = testingrf), obs = testingrf$Volume)
#   RMSE     Rsquared MAE 
#   151.82   0.9383   95.989

#Predict Output----
predicted= predict(rfFit, testingrf)
print(predicted)
str(predicted)
#not sure why this is helpful since the outcome is hard to 'read'

#SVM MODEL----
#svmLinear2 model: svmFit AUTOM GRID
#type: neighourhood based implicitly maps inputs to high-dimens feature spaces.
#package: e1071
#dataframe = final_df
#Y Value = Volume

set.seed(15)

#SET SPLIT 75%/25% for train/test in the dataset
inTrainingsvm <- createDataPartition(BWsample$Volume, p = .75, list = FALSE)
trainingsvm <- BWsample[inTraining,]
testingsvm <- BWsample[-inTraining,]

#10 fold cross validation
fitControlsvm <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

#train svm model with a tuneLenght = `1`
#  preProcess=c("center", "scale") removed because not appl on prod types
svmFit <- train(Volume~., 
               data = trainingsvm, 
               method = "svmLinear2", 
               trControl=fitControlsvm, 
               tuneLength = 1
)

#training results
svmFit
#SVM traning results----
#   RMSE     Rsquared  MAE Tuning par cost was held constant at value 0.25
#   1002.58   0.8955   517.39
saveRDS(object = svmFit, file = "svmFit.rds")

#SVM postresample----
postResample(pred = predict(object = svmFit, newdata = testingsvm), obs = testingsvm$Volume)
#   RMSE     Rsquared MAE 
#   410.01    0.6026  255.39

#Predict Output----
predicted= predict(svmFit, testingsvm)
print(predicted)
str(predicted)
#not sure why this is helpful since the outcome is hard to 'read'

#### REVIEW by PLOTS ####
#Model plot LM----
ggplot(data = objects(predicted), aes(x = ProductTypeNetbook, y = Volume)) +
  #geom_point() +
  #geom_smooth(method = "lm", se = FALSE)