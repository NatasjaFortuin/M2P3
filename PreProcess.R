#Libraries----
library(readr)
library(caret)
library(lattice)
library(ggplot2)
library (corrplot)
library(mlbench)

#Importdata----
Exist <- existingproductattributes2017

#Preprocessing----
# dummify the data
DummyVarsExist <- dummyVars(" ~ .", data = Exist)
readyData <- data.frame(predict(DummyVarsExist, newdata = Exist))

#View data----
str(existingproductattributes2017)
str(DummyVarsExist)
str(readyData)
View(readyData)
is.na(readyData)
sum(is.na(readyData))
head((readyData))
readyData$BestSellersRank <- NULL
sum(is.na(readyData))

# first selected relevant features----
relevant_vars <- c(
  "ProductTypeLaptop","ProductTypeNetbook","ProductTypePC",
  "ProductTypeSmartphone","ProductNum","x5StarReviews","x4StarReviews",
  "x3StarReviews","x2StarReviews","x1StarReviews","PositiveServiceReview",
  "NegativeServiceReview","Recommendproduct","Volume"
  )

# correlationplots----
corrData <- cor(readyData[relevant_vars])
corrplot(corrData, method ="ellipse")
corrplot(corrData, method = "number")
corrplot(corrData, method = "pie", type = "upper")
corrplot.mixed(corrData)
corrplot(readyData, type = "upper")
ggsave("correlationmatrixallattributes.png", width = 5, height = 5)
corrData <- cor(readyData)
corrplot

set.seed(7)
# load the dataset
readyData

# prepare training scheme----
control <- trainControl(method="repeatedcv", number=10, repeats=1)

#Rank Important attributes----
RankImp <- caret::train(Volume~., 
                 data=readyData[relevant_vars], 
                 method="rpart", # 1,
                 preProcess="center", 
                 trControl=control)
warnings()
corrDataImpAttributes = corrData
corrDataImpAttributes < - cor(corrData)
print(corrDataImpAttributes)
findCorrelation(corrDataImpAttributes, cutoff=0.9, verbose=FALSE, names=FALSE, exact = FALSE)
print(corrDataImpAttributes)
findcorrelationneg90 = corrDataImpAttributes
findCorrelation(findcorrelationneg90, cutoff=-0.9, verbose=FALSE, names=FALSE, exact = FALSE)
print(findcorrelationneg90)               
corrplot(findcorrelationneg90)
findCorrelation()
cor(corrDataImpAttributes)
names(findcorrelationneg90)
head(findcorrelationneg90)
corrplot(findcorrelationneg90)
corrplot(findcorrelationneg90, order = "hclust", addrect = 2)
ggsave("findcorrelationneg90.png", width = 5, height = 5)
findcorrelationneg90=corrData
corrplot(corrData)
corrplot(corrData, order = "hclust", addrect = 3)
corrplot(abs(corrData), order = "AOE", cl.lim = c(0,1))
ggsave("corrAOEOrder.png", width = 5, height = 5)
corrDataImpAttributes = corrData
SignCorr <- cor.mtest(corrDataImpAttributes, conf.level = .95)
print(SignCorr)
plot(SignCorr)
corrplot(SignCorr, p.mat = SignCorr$p, sig.level = 2)
corrplot(SignCorr, p.mat = SignCorr$p, insig = "blank" )

#Select final relevant attributes----
#based on VarImp(RankImp) rpart took all lower than 50 out because of overfitting (5*)
#and not correlated

Final_relevant_vars <- c(
  "ProductTypeLaptop","ProductTypeNetbook","ProductTypePC",
  "ProductTypeSmartphone","ProductNum","x4StarReviews",
  "x3StarReviews","1StarReviews", "PositiveServiceReview", "Volume")

corrData <- cor(readyData[relevant_vars])
corrplot(corrData, method ="ellipse")
corrplot(corrData, method = "number")
corrplot(corrData, method = "pie", type = "upper")
ggsave("correlationmatrixfinalattributes.png", width = 5, height = 5)
corrplot.mixed(corrData)
