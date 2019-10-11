#Libraries----
library(readr)
library(caret)
library(lattice)
library(ggplot2)

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
