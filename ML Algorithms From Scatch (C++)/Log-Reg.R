df <- read.csv("data/titanic_project.csv", header=TRUE)


df$pclass <- factor(df$pclass)
df$survived <- factor(df$survived)
attach(df)


# print metrics for accuracy, sensitivity, specificity

# Use the first 900 observations for train, the rest for test. 
set.seed(1234)
i <- sample(1:nrow(df), 900, replace=FALSE)
train <- df[i,]
test <- df[-i,]

# start the clock!
start_time <- Sys.time()

# train a Logistic Regression model on the train data, survived~pclass
glm1 <- glm(survived~pclass, data=train, family="binomial")

# stop the clock!
stop_time <- Sys.time()

# test on the test data
probs <- predict(glm1, newdata=test, type="response")
pred <- ifelse(probs>.5, 1, 0)
acc <- mean(pred==as.factor(test$survived))


run_time <- stop_time - start_time

# print the model summary to see the coefficients
summary(glm1)
detach(df)


# print metrics for accuracy, sensitivity, specificity
tab <- table(as.factor(pred), test$survived)
tab 

tp <- tab[1,1]
tn <- tab[2,2]
fp <- tab[1,2]
fn <- tab[2,1]

print(paste("accuracy = ", acc))
sens <- ((tp)/(tp+fn))
print(paste("sensitivity = ", sens))
spec <- ((tn)/(tn+fp))
print(paste("specificity = ", spec))

print(run_time)
