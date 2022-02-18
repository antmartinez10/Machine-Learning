df <- read.csv("data/titanic_project.csv", header=TRUE)

# make pclass, survived and sex factors
df$pclass <- factor(df$pclass)
df$survived <- factor(df$survived)
df$sex <- factor(df$age)

# split data, use first 900 observations as train, rest is test
set.seed(1234)
i <- sample(1:nrow(df), 900, replace=FALSE)
train <- df[i,]
test <- df[-i,]

# make sure there are no na's
sum(is.na(df$age))
sum(is.na(df$pclass))
sum(is.na(df$survived))
sum(is.na(df$sex))

# train a naïve Bayes model on the train data, survived~pclass+sex+age
library(e1071)

# start the clock!
start_time <- Sys.time()

nb1 <- naiveBayes(survived~pclass+sex+age, data=train)

# stop the clock!
stop_time <- Sys.time()

# print the model, which will show all the probabilities learned from the data
print(nb1)

# test on the test data
p1 <- predict(nb1, newdata=test, type="class")
table1 <- table(p1,test$survived)
print(table1)
# print metrics for accuracy, sensitivity, specificity

# used caret to confirm metrics
# library(caret)
# confusionMatrix(p1,test$survived)

tp <- table1[1,1]
tn <- table1[2,2]
fp <- table1[1,2]
fn <- table1[2,1]

run_time <- stop_time - start_time
print(paste("Run time is:", run_time))
acc <- mean(p1==test$survived)
print(paste("accuracy = ", acc))
sens <- ((tp)/(tp+fn))
print(paste("sensitivity = ", sens))
spec <- ((tn)/(tn+fp))
print(paste("specificity = ", spec))







