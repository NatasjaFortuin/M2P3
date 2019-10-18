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
  "ProductTypeSmartphone","ProductNum","x4StarReviews",
  "x3StarReviews","PositiveServiceReview","Volume"
)

# create correlation matrix----
cor(readyData[Final_relevant_vars])
corrplot(readyData)
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
# 0.8699                0.8467 
saveRDS(object = lmFit, file = "lmFit.rds")

#Predict Output----
predicted= predict(lmFit, testing_lm)
print(predicted)
str(predicted)

#Save predictions LM Model in separate column----
final_df$predLM <- predict(lmFit, testing_lm)

#LM postresample----
postResample(pred = predict(object = lmFit, newdata = testing_lm), obs = testing_lm$Volume)
##output =        RMSE   Rsquared    MAE
##lmFit =  +/-4354.847    0.5672     271.163


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
#   864.3071  0.8787   MAE 463.236

#KNN summary KNNFit K3----
summary(KNNFit)
#summaryperformance_KNNFit= Min Mean Abs Error: 420.7407, Min Mean S-error 1945 
saveRDS(object = KNNFit, file = "KNNFit.rds")

#KNN postresample----
postResample(pred = predict(object = KNNFit, newdata = testingKNN), obs = testingKNN$Volume)
#   RMSE     Rsquared MAE 
#   380.911  0.6624   210.75 

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
#   870.75   0.978     464.13
saveRDS(object = rfFit, file = "rfFit.rds")

#RF postresample----
postResample(pred = predict(object = rfFit, newdata = testingrf), obs = testingrf$Volume)
#   RMSE     Rsquared MAE 
#   143.58   0.955    101.373

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
#   787.177   0.9629   433.4216
saveRDS(object = svmFit, file = "svmFit.rds")

#SVM postresample----
postResample(pred = predict(object = svmFit, newdata = testingsvm), obs = testingsvm$Volume)
#   RMSE     Rsquared MAE 
#   392.461   0.5861  243.71

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
names(final_df)<-c("Laptop","Netbook","PC", "Phone", "ID", "x4Star", "x3Star", 
    "PosSerRev", "Volume", "predSVM", "predLM", "predKNN", "predRF")
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

Finaldf_cleaned <- distinct(.data = final_df_ExOut, PosSerRev, x4Star, x3Star, Volume, .keep_all = TRUE)
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
# 0.6784                0.6294
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
##lmFit =  +/-320.680  0.6456    223.876


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
#   266.3007  0.84267   MAE 159.5161

#KNN summary KNNFit2 K3----
summary(KNNFit2)
#summaryperformance_KNNFit= Min Mean Abs Error: 150.2986, Min Mean S-error 8705 
saveRDS(object = KNNFit2, file = "KNNFit2.rds")

#KNN postresample----
postResample(pred = predict(object = KNNFit2, newdata = testingKNN2), obs = testingKNN2$Volume)
#   RMSE     Rsquared MAE 
#   191.826   0.8312  102.92 

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
#   277.3768   0.8645   193.8815
saveRDS(object = rfFit2, file = "rfFit2.rds")

#RF postresample----
postResample(pred = predict(object = rfFit2, newdata = testingrf2), obs = testingrf2$Volume)
#   RMSE     Rsquared MAE 
#   137.120   0.9509  109.22

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
#   511.667  0.7061    281.878
saveRDS(object = svmFit2, file = "svmFit2.rds")

#SVM postresample----
postResample(pred = predict(object = svmFit2, newdata = testingsvm2), obs = testingsvm2$Volume)
#   RMSE     Rsquared  MAE 
#   286.367  0.63371   121.99

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
ggsave("Errorplot_LM.png", width = 5, height = 5)

PlotErrorCheck <- ggplot(data = PredData) +
  geom_point(aes(x = Volume, y = predLM)) +
  geom_abline(intercept = 1)

ggplot(data = PredData) +
  geom_point(aes(x = Volume, y = predKNN)) +
  geom_abline(intercept = 1)
ggsave("Errorplot_KNN.png", width = 5, height = 5)

ggplot(data = PredData) +
  geom_point(aes(x = Volume, y = predRF)) +
  geom_abline(intercept = 1)
ggsave("Errorplot_RF.png", width = 5, height = 5)

ggplot(data = PredData) +
  geom_point(aes(x = Volume, y = predSVM)) +
  geom_abline(intercept = 1)
ggsave("Errorplot_SVM.png", width = 5, height = 5)

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
ggsave("Smartphoneplot_LM.png", width = 5, height = 5)

#Model plot KNN----
ggplot(data = PredData, aes(x = Phone, y = predKNN)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)
ggsave("Smartphoneplot_KNN.png", width = 5, height = 5)

#Model plot RF----
ggplot(data = PredData, aes(x = Phone, y = predRF)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)
ggsave("Smartphoneplot_RF.png", width = 5, height = 5)

#Model plot SVM----
ggplot(data = PredData, aes(x = Phone, y = predSVM)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)
ggsave("SmartphoneplotRF.png", width = 5, height = 5)
