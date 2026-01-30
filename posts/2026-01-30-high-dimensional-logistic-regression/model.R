# Clean up environment and set working directory
rm(list=ls())
set.seed(128)

# Load the data
load("data_nir_tablets.RData")

# Load the required packages
library(Rtsne)
library(glmnet)

# Investigate the dimensions of the training and test data sets
dim(x)
length(y)
dim(x_test)
length(y_test)

# check how balanced the classes are in the training and test data
table(y)
table(y_test)

# Visualize the data with a t-distributed stochastic neightbour embedding plot
x_all <- rbind(x, x_test)
y_all <- c(y, y_test)
rtsne <- Rtsne(x_all, perplexity = 30)
colours <- c("red", "blue")[y_all+1]
plot(rtsne$Y, pch=19, col=adjustcolor(colours, 0.3), main="All Data - t-distributed stochastic neighbour embedded")
rtsne_train <- Rtsne(x, perplexity = 30)
colours_train <- c("red", "blue")[y+1]
plot(rtsne_train$Y, pch=19, col=adjustcolor(colours, 0.3), main="Training Data - t-distributed stochastic neighbour embedded")

# Standardize the data
x_stand <- scale(x, center=TRUE, scale=TRUE)
y_stand <- scale(y, center=TRUE, scale=TRUE)
x_test_stand <- scale(x_test, center=TRUE, scale=TRUE)
y_test_stand <- scale(y_test, center=TRUE, scale=TRUE)

# define the error function
classification_error <- function(y, yhat){
  tab <- table(y, yhat)
  error_rate <- (1 - (sum(diag(tab)) / (length(y))))
  error_rate
}

# define the threshold value tau
tau <- 0.5

# define the threshold function
assign_class <- function(probability){
  classes <- ifelse(probability > tau, 1, 0)
  classes
}

# define a function for running logistic regression
run_logistic_regression <- function (regularisation, num_lambda, training_iterations, validation_ratio){
  # Identify the Regularization option for printed output
  L_penalty_option <- ifelse(regularisation==1, "L1 Regularization", "L2 Regularization")
  legend_location <- ifelse(regularisation==1, "bottomright", "topright")
  
  # specify the range of values for lambda
  lambda <- seq(0.005, 0.150, length=num_lambda)
  
  # define vectors to store results
  error_train <- matrix(NA, training_iterations, num_lambda)
  error_val <- matrix(NA, training_iterations, num_lambda)
  error_test <- rep(NA, training_iterations)
  
  # define the portion of the training data to use for validation
  training_rows <- length(y)
  L <- floor(training_rows * validation_ratio)
  
  for (b in 1:training_iterations){
    # sample the training and validation data sets for this iteration
    val <- sample(1:training_rows, L)
    train <- setdiff(1:training_rows, val)
    
    # fit the logistic regression model
    fit <- glmnet(x[train, ], y[train], family="binomial", alpha=regularisation, lambda = lambda)
    
    # predict the classes for the training and validation data sets
    probabilities_train <- predict(fit, newx=x[train,], type="response")
    classes_train <- apply(probabilities_train, 2, assign_class)
    probabilities_val <- predict(fit, newx=x[val,], type="response")
    classes_val <- apply(probabilities_val, 2, assign_class)
    
    # calculate the misclassification rate
    error_train[b,] <- sapply(1:num_lambda, function(l) classification_error(y[train], classes_train[,l]))
    error_val[b,] <- sapply(1:num_lambda, function(l) classification_error(y[val], classes_val[,l]))
  }
  
  # plot the error rate against 1- lambda for the training and validation stage
  matplot(x = 1-lambda, t(error_val), type='l', lty=1, ylab='Error Rate', xlab='1 - lambda',
          col=adjustcolor("blue", 0.25), log="y",
          main="Plot of (1-lambda) versus error rate")
  
  matplot(x = 1-lambda, t(error_train), type='l', lty=1, 
          col=adjustcolor("black", 0.25), log="y", add=TRUE)
  
  lines(1-lambda, colMeans(error_train), col = "black", lwd = 2)
  
  lines(1-lambda, colMeans(error_val), col = "blue", lwd = 2)
  
  legend(legend_location, legend = c("Training error", "Validation error", "Optimal Lambda"),
         fill = c("black", "blue", "red"), bty = "n")
  
  # identify the value of lambda that yielded the lowest error rate for the validation data
  lambda_best_val <- lambda[which.min(colMeans(error_val))]
  cat("\n", L_penalty_option,  ": The value of lambda that results in the lowest error rate for the validation data is: ", lambda_best_val)
  
  # plot the optimal value for 1 - lambda i.e. the optimal model complexity
  abline(v = 1 - lambda_best_val, col="red", lwd=2)
  
  # identify the value of lambda that yielded the lowest error rate for the training data
  lambda_best_train <- lambda[which.min(colMeans(error_train))]
  cat("\n", L_penalty_option , ": The value of lambda that results in the lowest error rate for the training data is: ", lambda_best_train)
  
  # Next use the optimal value of lambda from the validation data to predict the class of the test data
  fit_test <- glmnet(x, y, family='binomial', lambda=lambda_best_val, alpha=regularisation)
  probabilities_test <- predict(fit_test, newx=x_test, type="response")
  classes_test <- apply(probabilities_test, 2, assign_class)
  
  # Compare the predicted classes with the actual test classes
  cat("\n", L_penalty_option , "The confusion matrix is:", "\n", "\n")
  print(table(y_test, classes_test))
  
  # Calculate the misclassification rate for the test data
  cat("\n", L_penalty_option , ": The missclassification rate for the test data is: ", 
      classification_error(y_test, classes_test))
  
  # The L1 penalty term should have reduced the number of x variables used in the model
  # while the L2 penalty term should have shrunk the coefficient values towards sero
  # Plot the values of the beta coefficients for all explanatory variables in x
  if(regularisation == 1){
    beta_coefficients <- coef(fit_test, s = lambda_best_val)
    cat("\n", L_penalty_option, ": The features selected by the L1 Regularization are at index: ", 
        which(beta_coefficients != 0))
    beta_coefficients_colours <- ifelse(beta_coefficients == 0, "black", "blue")
    plot(beta_coefficients, col = beta_coefficients_colours, pch=19, xlab="Beta Coefficients", ylab="Estimated Coefficient",
         main="Plot of feature selection by L1 Regularization")
  } else {
    beta_coefficients <- coef(fit_test, s = lambda_best_val)
    which(beta_coefficients > 0.01)
    beta_coefficients_colours <- ifelse(beta_coefficients < 0.01, "black", "blue")
    plot(beta_coefficients, col = beta_coefficients_colours, pch=19, xlab="Beta Coefficients", ylab="Estimated Coefficient",
         main="Plot of feature shrinkage by L2 Regularization")
  }
  
  cat("\n", "End of regression run", "\n", "\n")
}

# Run logistic regression algorithm with L1 Regularization (Lasso Regression)
run_logistic_regression(regularisation=1, num_lambda=100, training_iterations=100, validation_ratio=0.3)

# Run logistic regression algorithn with L2 Regularization (Ridge Regression)
run_logistic_regression(regularisation=0, num_lambda=100, training_iterations=100, validation_ratio=0.3)