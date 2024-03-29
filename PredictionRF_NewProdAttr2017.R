#Libraries----
library(readr)
library(caret)
library(lattice)
library(ggplot2)
library (corrplot)
library(mlbench)
library(dplyr)

#Importdata----
newproductattributes2017 <- read_csv("newproductattributes2017.csv")

#Preprocessing----
# dummify the data
DummyVarsNew <- dummyVars(" ~ .", data = newproductattributes2017)
NewData <- data.frame(predict(DummyVarsNew, newdata = newproductattributes2017))
saveRDS(object = DummyVarsNew, file = "DummyVarsNew.rds")
readRDS("DummyVarsNew.rds")

#View data----
str(NewData)
is.na(NewData)
sum(is.na(NewData))
head((NewData))
NewData$BestSellersRank <- NULL
sum(is.na(NewData))

head(NewData)

#Rename columnnames NewData----
names(NewData)<-c("Acc", "Disp", "ExWar", "GameC", "Laptop","Netbook","PC", "Printer", "PrSupp", "Phone", "SW", "Tablet", "ID", "Price", "x5Star", "x4Star","x3Star", "x2Star", "x1Star", "PosSerRev", "NegSerRev", "RecProd", "Weight", "Depth", "Width", "Height", "PMargin", "Volume")
head(NewData)

#Remove irrelevant Producttypes from the data
NewData <- select (NewData, -c(Acc,Disp,ExWar,GameC,Printer,PrSupp,SW,Tablet))
head(NewData)

#### PREDICT ####
#rfFit2 BEST PERFOM MOD----
NewData$PredVol <- predict(object = rfFit2, newdata=NewData)

#Postpresample---- CHECK KPI'S of new column
postResample(pred = NewData$PredVol, obs = NewData$PredVol)

head(NewData)

summary(NewData)

str(NewData)

#RF Pred Plot----
#Netbook----
ggplot(data = NewData, aes(x = Netbook, y = PredVol)) +
  geom_jitter(width = 5, height = 5)

PlotNetbook <- ggplot(data = NewData, aes(x = Netbook, y = PredVol)) +
  geom_jitter(width = 5, height = 5)

#Laptop----
ggplot(data = NewData, aes(x = Laptop, y = PredVol)) +
  geom_jitter(width = 5, height = 5)

PlotLaptop <- ggplot(data = NewData, aes(x = Laptop, y = PredVol)) +
  geom_jitter(width = 5, height = 5)

#PC----
ggplot(data = NewData, aes(x = PC, y = PredVol)) +
  geom_jitter(width = 5, height = 5)

PlotPC <- ggplot(data = NewData, aes(x = PC, y = PredVol)) +
  geom_jitter(width = 5, height = 5)

#Smartphone----
ggplot(data = NewData, aes(x = Phone, y = PredVol)) +
  geom_jitter(width = 5, height = 5)

PlotPhone <- ggplot(data = NewData, aes(x = Phone, y = PredVol)) +
  geom_jitter(width = 5, height = 5)

##############n
#KNNFit2----
NewData$PredVolKNN <- predict(object = KNNFit2, newdata=NewData)

#KNN Pred Plot----
#Netbook----
ggplot(data = NewData, aes(x = Netbook, y = PredVolKNN)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)
#Laptop----
ggplot(data = NewData, aes(x = Laptop, y = PredVolKNN)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)
#PC----
ggplot(data = NewData, aes(x = PC, y = PredVolKNN)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)
#Smartphone----
ggplot(data = NewData, aes(x = Phone, y = PredVolKNN)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#lmFit2----
NewData$PredVolLM <- predict(object = lmFit2, newdata=NewData)

#LM Pred Plot----
#Netbook----
ggplot(data = NewData, aes(x = Netbook, y = PredVolLM)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)
#Laptop----
ggplot(data = NewData, aes(x = Laptop, y = PredVolLM)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)
#PC----
ggplot(data = NewData, aes(x = PC, y = PredVolLM)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)
#Smartphone----
ggplot(data = NewData, aes(x = Phone, y = PredVolLM)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

#svmFit2----
NewData$PredVolSVM <- predict(object = svmFit2, newdata=NewData)

#SVM Pred Plot----
#Netbook----
ggplot(data = NewData, aes(x = Netbook, y = PredVolSVM)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)
#Laptop----
ggplot(data = NewData, aes(x = Laptop, y = PredVolSVM)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)
#PC
ggplot(data = NewData, aes(x = PC, y = PredVolSVM)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)
#Smartphone----
ggplot(data = NewData, aes(x = Phone, y = PredVolSVM)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

View(NewData)


write.csv(NewData, file="PredictedData.csv", row.names = TRUE)
