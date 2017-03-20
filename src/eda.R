#load the project
library('ProjectTemplate')
load.project()

#load libraries

library(caret)
library(randomForest)
library(gbm)
library(rattle)
library(pROC)
library(plyr)
library(e1071)
library(fBasics)
library(Hmisc)
library(base)
library(tidyr)
library(ggplot2)

#use the variable wbcd to manipulate the dataset

wbcd <- breast.cancer.wisconsin

#------------------------------------------------#
#exploratory data analysis                       #
#------------------------------------------------#

dim(wbcd) #number of rows and columns

summary(wbcd) #6 number summary

colnames(wbcd)[colSums(is.na(wbcd)) > 0] #find which column contains missing values

sapply(wbcd,skewness) #skewness

sapply(wbcd,kurtosis) #kurtosis

#histogram plots

wbcd <- wbcd[, -1] %>%
  gather(measure, value, radius_mean:fractal_dimension_worst)

ggplot(data = wbcd, aes(x = value, fill = diagnosis, color = diagnosis)) +
  geom_density(alpha = 0.3, size = 1) +
  geom_rug() +
  scale_fill_brewer(palette = "Set2") +
  scale_color_brewer(palette = "Set2") +
  facet_wrap( ~ measure, scales = "free_y", ncol = 3)

#------------------------------------------------#
#data transformation                             #    
#------------------------------------------------#

#impute id, X 

wbcd$id <- NULL

wbcd$X <- NULL

#assign Benign, Malignant to diagnosis and change type to factor

wbcd$diagnosis <- factor(wbcd$diagnosis, levels = c("B", "M"),
                         labels = c("Benign", "Malignant"))

#verify transformation 

str(wbcd)

#check number of observations and percentage

cbind(Frequency = table(wbcd$diagnosis), Percentage = (prop.table(table(wbcd$diagnosis))* 100))

#data summary

summary(wbcd)

#------------------------------------------------#
#Build the model                                 #
#------------------------------------------------#

#ensure reproducibility with set.seed()

set.seed(1)

#build training/testing sets, 70% training, 30% testing

inTrain <- createDataPartition(wbcd$diagnosis, p=0.7, list=FALSE)

training <- wbcd[inTrain,]

testing <- wbcd[-inTrain,]

dim(training) #size of training set

dim(testing) #size of testing set

#outcome distribution in the testing set

table(testing$diagnosis)

#build a random forest model using default parameters

set.seed(1)

rf_model <- train(diagnosis ~ ., data=training, method="rf")

rf_model

rf_model$finalModel

#performance on testing test

pred_test <- predict(rf_model, testing)

confusionMatrix(pred_test, testing$diagnosis, positive = "Malignant")

#-----------------------------------------------------#
#Model tuning                                         #
#-----------------------------------------------------#

#tune Rf model to find best mtry

#using cross-validation

ctrl <- trainControl(method="repeatedcv", repeats=3)

grid <- expand.grid(mtry = c(1, 2, 3, 5, 7, 10, 15, 20, 30))

#using expanded grid

set.seed(1)

rf_tune_model <- train(diagnosis ~ ., data=training, method="rf", tuneGrid = grid, trControl = ctrl)

rf_tune_model

rf_tune_model$finalModel

plot(rf_tune_model)

#performance of updated model on testing set

pred_tuned_test <- predict(rf_tune_model, testing)

confusionMatrix(pred_tuned_test, testing$diagnosis, positive="Malignant")

#----------------------------------------------------------#
#Other performance measure                                 #
#----------------------------------------------------------#

#variable importance

rf_var <- varImp(rf_tune_model)

plot(rf_var)

#ROC curve
rf_roc <- roc(testing$diagnosis,
              predict(rf_tune_model, testing, type = "prob")[,"Malignant"],
              levels = rev(levels(testing$diagnosis)))

roc_value <- auc(rf_roc)

#plot ROC curve

plot.roc(smooth(rf_roc), main = "RF ROC Curve", col = "blue")

legend("bottom", legend = roc_value, col=c("#1c61b6"), lwd=2)

#for reproducibility purposes

sessionInfo()

