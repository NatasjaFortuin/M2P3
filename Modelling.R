library(readr)
library(caret)
library(ggplot2)
library(mlbench)
library(e1071)
library(dplyr)

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

#Save predictions LM Model in separate column----
final_df$predLM <- predict(lmFit, testing_lm)

#LM postresample----
postResample(pred = predict(object = lmFit, newdata = testing_lm), obs = testing_lm$Volume)
##output = RMSE   Rsquared    MAE
##lmFit =  +/-434.867  0.5694    273.809


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

#Save predictions KNN Model in separate column----
final_df$predKNN <- predict(KNNFit, testingKNN)

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

#Save predictions RF Model in separate column----
final_df$predRF <- predict(rfFit, testingrf)

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

#Save predictions SVM Model in separate column----
final_df$predSVM <- predict(svmFit, testingsvm)

str(final_df)
View(final_df)

#create excel----
write.csv(final_df, file = "ExistVolumeInclPred", row.names = TRUE)


#### REVIEW by PLOTS ####

#NETBOOK----

#Model plot LM----
ggplot(data = final_df, aes(x = ProductTypeNetbook, y = predLM)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#Model plot KNN----
ggplot(data = final_df, aes(x = ProductTypeNetbook, y = predKNN)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#Model plot RF----
ggplot(data = final_df, aes(x = ProductTypeNetbook, y = predRF)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#Model plot SVM----
ggplot(data = final_df, aes(x = ProductTypeNetbook, y = predSVM)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#LAPTOP----

#Model plot LM----
ggplot(data = final_df, aes(x = ProductTypeLaptop, y = predLM)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#Model plot KNN----
ggplot(data = final_df, aes(x = ProductTypeLaptop, y = predKNN)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#Model plot RF----
ggplot(data = final_df, aes(x = ProductTypeLaptop, y = predRF)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#Model plot SVM----
ggplot(data = final_df, aes(x = ProductTypeLaptop, y = predSVM)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#PC----

#Model plot LM----
ggplot(data = final_df, aes(x = ProductTypePC, y = predLM)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#Model plot KNN----
ggplot(data = final_df, aes(x = ProductTypePC, y = predKNN)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#Model plot RF----
ggplot(data = final_df, aes(x = ProductTypePC, y = predRF)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#Model plot SVM----
ggplot(data = final_df, aes(x = ProductTypePC, y = predSVM)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#SMARTPHONE----

#Model plot LM----
ggplot(data = final_df, aes(x = ProductTypeSmartphone, y = predLM)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#Model plot KNN----
ggplot(data = final_df, aes(x = ProductTypeSmartphone, y = predKNN)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#Model plot RF----
ggplot(data = final_df, aes(x = ProductTypeSmartphone, y = predRF)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#Model plot SVM----
ggplot(data = final_df, aes(x = ProductTypeSmartphone, y = predSVM)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#Rename columnnames final_df----
head(final_df)
names(final_df)<-c("Laptop","Netbook","PC", "Phone", "ID", "x1Star", "x4Star", "x3Star", "PosSerRev", "Volume", "predSVM", "predLM", "predKNN", "predRF")
head(final_df)

#Find outliers
outlier_values <- boxplot.stats(final_df$Volume)$out
boxplot(final_df$Volume)
boxplot(final_df$predSVM)
boxplot(final_df$predLM)
boxplot(final_df$predKNN)
boxplot(final_df$predRF)
boxplot(final_df$Volume)$out
#outliers determined as values 7036 and 11204
#find in which row the outliers are
final_df[which(final_df$Volume %in% outlier_values),]
#outliers are in rows 50 (11204) and 73 (7036)

#ERROR Check ----
#Error check is done with Volume & Pred Volume!!
ggplot(data = final_df) +
  geom_point(aes(x = Volume, y = predRF)) +
  geom_abline(intercept = 1)

View(outliers_values)

#Remove Outliers----
final_df_ExOut <- final_df[-which(final_df$Volume %in% outlier_values),]
#check removal with boxplot
boxplot(final_df_ExOut)
boxplot(final_df_ExOut$Volume)

#Remove Duplicates----
duplicated(final_df_ExOut$Volume)
duplicates <- duplicated(final_df_ExOut$Volume)
final_df_ExOut[which (final_df_ExOut$Volume %in% duplicates),]
#didn't select the duplicates from prod id 135 tm 141 that I want to remove
#so nothing removed yet

duplicated(final_df_ExOut$PosSerRev)
duplicates2 <- duplicated(final_df_ExOut$PosSerRev)
final_df_ExOut[which (final_df_ExOut$PosSerRev %in% duplicates),]
#works with normalized PosServRev 0-1 so is of no use. Nothing removed.
#doesn't work because duplicate values are not recognized as such
#I know it is product ID's 134 tm 141. I want to keep 134 and remove rest

Finaldf_cleaned <- final_df_ExOut[!(final_df_ExOut$ID==135  
                                    final_df_ExOut$ID==136
                                    final_df_ExOut$ID==137 
                                    final_df_ExOut$ID==138 
                                    final_df_ExOut$ID==139 
                                    final_df_ExOut$ID=140 
                                    final_df_ExOut$ID=141),]
#not working. Tried +, , AND etc...

subset(Finaldf_cleaned, ID!=135)
subset(Finaldf_cleaned, ID!=136)
subset(Finaldf_cleaned, ID!=137)
subset(Finaldf_cleaned, ID!=138)
subset(Finaldf_cleaned, ID!=139)
subset(Finaldf_cleaned, ID!=140)
subset(Finaldf_cleaned, ID!=141)

View(Finaldf_cleaned)
#did not work, only removes one value at a time. Replace dfcleaned with 
#original dataframe ex Outliers

Finaldf_cleaned <- final_df_ExOut
str(Finaldf_cleaned)

Finaldf_cleaned <- distinct(.data = final_df_ExOut, PosSerRev, x1Star, x4Star, x3Star, Volume, .keep_all = TRUE)
rm(Finalsdf_cleaned)
str(Finaldf_cleaned)
View(Finaldf_cleaned)

#remove prediction colums with dplyr in order to re run the modelling
Finaldf_cleaned <- select (Finaldf_cleaned, -c(predLM, predRF, predKNN, predSVM))

View((Finaldf_cleaned))
set.seed(15)

#create a 20% sample of the data----
BWsample2 <- Finaldf_cleaned[sample(1:nrow(Finaldf_cleaned), 70,replace=FALSE),]

# define an 75%/25% train/test split of the dataset----
inTraining_lm2 <- createDataPartition(BWsample2$Volume, p = .75, list = FALSE)
training_lm2 <- BWsample2[inTraining,]
testing_lm2 <- BWsample2[-inTraining,]

#### MODELLING CLEANED ####

#LINEAR MODEL----
#lm model: lmfFit2 AUTOM GRID
#type: line y based on x model
#package: baseR 
#dataframe = Finaldf_cleaned
#Y Value = Volume

lmFit2 <- lm(Volume~., 
            data = training_lm2)

#training results
lmFit2
saveRDS(lmFit2, file = "lmFit2.rds")
#LM summary lmFit----
summary(lmFit2)
#summaryperformance_lmFit2
#multiple R-squared   Adjusted R-squared 
# 0.6948                0.6324
saveRDS(object = lmFit2, file = "lmFit2.rds")

#Predict Output----
predicted= predict(lmFit2, testing_lm2)
print(predicted)
str(predicted)

#Save predictions LM Model in separate column----
Finaldf_cleaned$predLM <- predict(lmFit2, Finaldf_cleaned)

#LM postresample----
postResample(pred = predict(object = lmFit2, newdata = testing_lm2), obs = testing_lm2$Volume)
##output =    RMSE   Rsquared    MAE
##lmFit =  +/-326.676  0.6156    215.198


#KNN MODEL----
#K-nn model: KNNfFit2 AUTOM GRID
#type: neighbour based model
#package: caret
#dataframe = Finaldf_Cleaned
#Y Value = Volume

set.seed(15)

#SET SPLIT 75%/25% for train/test in the dataset
inTrainingKNN2 <- createDataPartition(BWsample2$Volume, p = .75, list = FALSE)
trainingKNN2 <- BWsample2[inTraining,]
testingKNN2 <- BWsample2[-inTraining,]

#10 fold cross validation
fitControlKNN2 <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

#train knn model with a tuneLenght = `1`(trains with 1 mtry values for knn)
#  preProcess=c("center", "scale") removed because not appl on prod types
KNNFit2 <- train(Volume~., 
                data = trainingKNN2, 
                method = "kknn", 
                trControl=fitControlKNN2, 
                tuneLength = 1
)

#training results
KNNFit2
#KNN traning results----
#   RMSE     Rsquared  MAE 
#   335.4799  0.7253143   MAE 209.0681

#KNN summary KNNFit2 K3----
summary(KNNFit2)
#summaryperformance_KNNFit= Min Mean Abs Error: 468.16, Min Mean S-error 1914 
saveRDS(object = KNNFit2, file = "KNNFit2.rds")

#KNN postresample----
postResample(pred = predict(object = KNNFit2, newdata = testingKNN2), obs = testingKNN2$Volume)
#   RMSE     Rsquared MAE 
#   196.067   0.8338  111.94 

#Predict Output----
predicted= predict(KNNFit2, testingKNN2)
print(predicted)
str(predicted)

#Save predictions KNN Model in separate column----
Finaldf_cleaned$predKNN <- predict(KNNFit2, Finaldf_cleaned)

#RF MODEL----
#Random Forest model: rfFit2 AUTOM GRID
#type: decision tree for mean prediction of individual trees
#package: caret
#dataframe = Finaldf_cleaned
#Y Value = Volume

set.seed(15)

#SET SPLIT 75%/25% for train/test in the dataset
inTrainingrf2 <- createDataPartition(BWsample2$Volume, p = .75, list = FALSE)
trainingrf2 <- BWsample2[inTraining,]
testingrf2 <- BWsample2[-inTraining,]

#10 fold cross validation
fitControlrf2 <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

#train knn model with a tuneLenght = `1`(trains with 1 mtry values for knn)
#  preProcess=c("center", "scale") removed because not appl on prod types
rfFit2 <- train(Volume~., 
               data = trainingrf2, 
               method = "rf", 
               trControl=fitControlrf2, 
               tuneLength = 1
)

#training results
rfFit2
#RF traning results----
#   RMSE     Rsquared  MAE 
#   269.39   0.8622    174.71
saveRDS(object = rfFit2, file = "rfFit2.rds")

#RF postresample----
postResample(pred = predict(object = rfFit2, newdata = testingrf2), obs = testingrf2$Volume)
#   RMSE     Rsquared MAE 
#   119.11   0.9480   74.09

#Predict Output----
predicted= predict(rfFit2, testingrf2)
print(predicted)
str(predicted)

#Save predictions RF Model in separate column----
Finaldf_cleaned$predRF <- predict(rfFit2, Finaldf_cleaned)

#SVM MODEL----
#svmLinear2 model: svmFit2 AUTOM GRID
#type: neighourhood based implicitly maps inputs to high-dimens feature spaces.
#package: e1071
#dataframe = Finaldf_cleaned
#Y Value = Volume

set.seed(15)

#SET SPLIT 75%/25% for train/test in the dataset
inTrainingsvm2 <- createDataPartition(BWsample2$Volume, p = .75, list = FALSE)
trainingsvm2 <- BWsample2[inTraining,]
testingsvm2 <- BWsample2[-inTraining,]

#10 fold cross validation
fitControlsvm2 <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

#train svm model with a tuneLenght = `1`
#  preProcess=c("center", "scale") removed because not appl on prod types
svmFit2 <- train(Volume~., 
                data = trainingsvm2, 
                method = "svmLinear2", 
                trControl=fitControlsvm2, 
                tuneLength = 1)

#training results
svmFit2
#SVM traning results----
#   RMSE     Rsquared  MAE Tuning par cost was held constant at value 0.25
#   588.6089  0.6468   347.0124
saveRDS(object = svmFit2, file = "svmFit2.rds")

#SVM postresample----
postResample(pred = predict(object = svmFit2, newdata = testingsvm2), obs = testingsvm2$Volume)
#   RMSE     Rsquared MAE 
#   294.894  0.5832   132.946

#Predict Output----
predicted= predict(svmFit2, testingsvm2)
print(predicted)
str(predicted)

#Save predictions SVM Model in separate column----
Finaldf_cleaned$predSVM <- predict(svmFit2, Finaldf_cleaned)

str(Finaldf_cleaned)
View(Finaldf_cleaned)

#### IMPROVE DATAFRAME FOR PLOTS ####
PredData <- Finaldf_cleaned
as.integer(PredData$predLM)
as.integer(PredData$predKNN)
as.integer(PredData$predRF)
as.integer(PredData$predSVM)

View(PredData)

#ERROR Check ----
#Error check is done with Volume & Pred Volume!!
ggplot(data = PredData) +
  geom_point(aes(x = Volume, y = predLM)) +
  geom_abline(intercept = 1)

ggplot(data = PredData) +
  geom_point(aes(x = Volume, y = predKNN)) +
  geom_abline(intercept = 1)

ggplot(data = PredData) +
  geom_point(aes(x = Volume, y = predRF)) +
  geom_abline(intercept = 1)

ggplot(data = PredData) +
  geom_point(aes(x = Volume, y = predSVM)) +
  geom_abline(intercept = 1)

#### REVIEW by PLOTS2 ####

#NETBOOK----

#Model plot LM----
ggplot(data = PredData, aes(x = Netbook, y = predLM)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE)

#Model plot KNN----
ggplot(data = PredData, aes(x = Netbook, y = predKNN)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE)

#Model plot RF----
ggplot(data = PredData, aes(x = Netbook, y = predRF)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE)

#Model plot SVM----
ggplot(data = PredData, aes(x = Netbook, y = predSVM)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#LAPTOP----

#Model plot LM----
ggplot(data = PredData, aes(x = Laptop, y = predLM)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#Model plot KNN----
ggplot(data = PredData, aes(x = Laptop, y = predKNN)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#Model plot RF----
ggplot(data = PredData, aes(x = Laptop, y = predRF)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#Model plot SVM----
ggplot(data = PredData, aes(x = Laptop, y = predSVM)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#PC----

#Model plot LM----
ggplot(data = PredData, aes(x = PC, y = predLM)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#Model plot KNN----
ggplot(data = PredData, aes(x = PC, y = predKNN)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#Model plot RF----
ggplot(data = PredData, aes(x = PC, y = predRF)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#Model plot SVM----
ggplot(data = PredData, aes(x = PC, y = predSVM)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#SMARTPHONE----

#Model plot LM----
ggplot(data = PredData, aes(x = Phone, y = predLM)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#Model plot KNN----
ggplot(data = PredData, aes(x = Phone, y = predKNN)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#Model plot RF----
ggplot(data = PredData, aes(x = Phone, y = predRF)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#Model plot SVM----
ggplot(data = PredData, aes(x = Phone, y = predSVM)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)
