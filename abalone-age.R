# https://archive.ics.uci.edu/ml/datasets/Abalone
library(gmodels)
library(randomForest)
library(e1071)
library(caret)
library(doSNOW)
library(vcd)
library(ROCR)

rm(list = ls())
setwd("C:\\Users\\Vette\\Desktop\\Regis\\MSDS680\\Week 7")
getwd()

# read the data
df <- read.csv('abalone.data', header = FALSE, stringsAsFactors=TRUE)

col.names <- c('Sex', 'Length', 'Diameter', 'Height', 'Whole_Weight', 'Shucked_Weight',
               'Viscera_Weight', 'Shell_Weight', 'Rings')

colnames(df) <- col.names

# move target variable to front of data frame, makes test/train split easier
# especially if features are added or deleted
Age <- df$Rings + 1.5
df$Rings <- NULL
df <- cbind(Age, df)

#plot(df$Diameter, df$Whole_Weight)
#plot(df$Length, df$Whole_Weight)
#plot(df$Age, df$Whole_Weight)
#plot(df$Age, df$Diameter)
#plot(df$Age, df$Sex)
#plot(df$Age, df$Length)
#plot(df$Age, df$Shell_Weight)


# ~ 7-13: 84-85% accurate
df$Age <- as.factor(ifelse(df$Age <= 7, 'Young', ifelse(df$Age <= 13, 'Adult', 'Old')))
table(df$Age)

#df$Sex <- NULL # RF a tiny bit worse, SVM a tiny bit better

# train / test split 70:30
set.seed(319)
labels <- sort(sample(nrow(df), nrow(df)*.7))
trainX <- df[labels,-1]
testX <- df[-labels,-1]
trainY <- df[labels,1]
testY <- df[-labels,1]
train.nn <- df[labels, ]
test.nn <- df[-labels, ]

set.seed(5556)
rf.mdl <- randomForest(trainY~., trainX, ntree=400, mtry=5,
                       keep.forest=TRUE, importance=TRUE,test=testX) # 
pred <- predict(rf.mdl, newdata=testX, type='response')
CrossTable(x = testY, y = pred, prop.chisq = FALSE)
misClasificError <- mean(pred != testY)
print(paste('Accuracy',(1-misClasificError)*100))
Kappa(table(testY, pred))

confusionMatrix(pred, testY, positive='Adult')
rf.imp <- importance(rf.mdl)
rf.imp
varImpPlot(rf.mdl)

library(rfUtilities)
accuracy(x = testY, y = pred)

# cross validation and training for random forest
# recomends mtry=5
# https://www.youtube.com/watch?v=84JSk36og34
# Langer video #5
set.seed(2348)
cv.10.folds <- createMultiFolds(trainY, k = 10, times = 10)
ctrl.1 <- trainControl(method = "repeatedcv", number = 10, repeats = 10,
                       index = cv.10.folds)
cl <- makeCluster(3, type = "SOCK")
registerDoSNOW(cl)
set.seed(34324)
rf.cv.1 <- train(x = trainX, y = trainY, method = "rf", tuneLength = 5,
                   ntree = 400, trControl = ctrl.1)
stopCluster(cl)
rf.cv.1


svm.mdl <- svm(trainY~. , trainX, kernel='radial', cost=20, gamma=.125)
pred <- predict(svm.mdl, newdata=testX, type='response')
#CrossTable(x=testY, y=pred, prop.chisq = FALSE)
misClasificError <- mean(pred != testY)
print(paste('Accuracy',(1-misClasificError)*100))
Kappa(table(testY, pred))
confusionMatrix(pred, testY, positive='Adult')

# tune the SVM
table(testY)

tuned <- tune.svm(Age~., data=train.nn, cost=c(1,5,10,20,30), gamma=c(0.0625,0.125,0.25),
                  kernel='polynomial', degree=c(2,3,4),
                  tune.control(cross=10))
tuned <- tune.svm(Age~., data=train.nn, cost=c(1,5,10,20,30), gamma=c(0.0625,0.125,0.25,0.375),
                  kernel='radial',
                  tune.control(cross=10))
summary(tuned)
