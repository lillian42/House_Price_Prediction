#####Coding for the final project#####
# Load Packages & Read Data
library(Metrics)
library(Matrix)
library(methods)
library(leaps)
library(ISLR)
library(corrplot)
library(glmnet)
library(list)
Houseprice <- read.csv("train.csv", stringsAsFactors = T)

### Data cleansing
# Using sapply to extract numerical data out of the dataset
Houseprice <- Houseprice[ ,sapply(Houseprice, is.numeric)]
# Replace NA with zero directly
Houseprice <- na.omit(Houseprice)


### Descriptive Analysis
# We have noticed that some values have strong correlation with saleprice, and here is an example of the overall quality correlates with its sale price
boxplot(SalePrice ~ OverallQual, Houseprice)
# Draw a corrplot of all numeric variables
correlation1 <- cor(Houseprice[,-1],use="everything")
corrplot(correlation1, method="ellipse", type="lower",  sig.level = 0.01, insig = "blank")
# The 3 pairs of variables "TotalBsmtSF" and "X1stFlrSF",  "TotRmsAbvGrd" and "GrLivArea", "GarageArea" and "GarageCars" have notable positive correlation with each other.
# "OverallQual", "GrLivArea", "GarageArea" and "GarageCars" have relative stronger correlations with SalesPrice
# Hence, plot the correlation among these four variables and SalePrice
pairs(~ SalePrice + OverallQual + TotalBsmtSF + GarageCars + GarageArea, data = Houseprice)
# The dependent variable (SalePrice) looks having decent linearity when plotting with other variables
# However, the problem of multicollinearity is obvious 


### Linear Regression
# Split Houseprice dataset to Training and Test subset with a ratio of 3:1
Training1<- Houseprice[1:floor(length(Houseprice[,1])*0.75),]
Test1<- Houseprice[(length(Training1[,1])+1):1460,]
# Put all variables into the regression model
reg1 <- lm(SalePrice ~. -Id, data = Training1)
summary(reg1)
# Manually pick out the variables which are more significant and remove the rest, and got modified version of linear regression model1.
reg1_Modified_1<-lm(formula = SalePrice ~ LotArea + OverallQual + OverallCond + YearBuilt + BsmtFinSF1 +
                      BsmtUnfSF + X1stFlrSF + X2ndFlrSF + BedroomAbvGr + KitchenAbvGr + TotRmsAbvGrd + GarageYrBlt, 
                    data = Training1)
summary(reg1_Modified_1)
# Check the performance of linear regression model with RMSE value, using function to obtain train and test RMSE for a fitted model.
get_rmse = function(model, data, response) {
  rmse(actual = data[, response], 
       predicted = predict(model, data))
}
get_rmse(model = reg1_Modified_1, data = Training1, response = "SalePrice")
get_rmse(model = reg1_Modified_1, data = Test1, response = "SalePrice")


### Model Selection
## Best Model Selection using exhaustive method
# Noting that "OverallQual" is a significant variable, we forced it in
subset1 <- regsubsets(x = SalePrice ~.-Id, data = Houseprice, nvmax=36, force.in = "OverallQual", method = "exhaustive")
summary(subset1)
summary1 <- summary(subset1)
plot(summary1$cp, xlab="Number of predictors", ylab="Cp")
plot(subset1, scale = "adjr2", main = "Adjusted R^2")
which.min(summary1$cp)
which.max(summary1$adjr2)
# Select the subset of variables of 19
coef(subset1, 19)

## LASSO
# The data frame x hold the data for predictors
x = model.matrix(SalePrice ~.-Id, data = Houseprice)[,-1]
# The vector y hold the data for the response, Saleprice in this case
y = Houseprice$SalePrice
#  10-fold cross-validation on the whole data to determine the best lambda value
set.seed(1)
cv.out = cv.glmnet(x, y, alpha = 1, type.measure = "mse")
plot(cv.out)
bestlam <- cv.out$lambda.1se
bestlam

# Ran ridge regression again with all our data and using the best lambda to obtain coefficient estimates
lasso.final = glmnet(x, y, alpha = 1, lambda = bestlam)
coef(lasso.final)

# XGBoost
suppressMessages({
  library(xgboost)
  library(foreach)    
})

# read train and test data set
dd <- read.csv("../input/train.csv", stringsAsFactors = FALSE)
dt <- read.csv("../input/test.csv", stringsAsFactors = FALSE)
dt$SalePrice <- NA # add sales price column to the train set to make it possible to merge both data sets
# merge both data sets to ensure all factor levels are the same
d <- rbind(dd,dt) 
# in the merged data set this is the train data, the data with known outcome
train_data_set = which(!is.na(d$SalePrice))
categorical <- c(
  "Id","MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", 
  "LotConfig","LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", 
  "HouseStyle", "RoofStyle", "RoofMatl","Exterior1st", "Exterior2nd", "MasVnrType", 
  "Foundation", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "CentralAir", 
  "Electrical", "Functional", "GarageType", "GarageFinish", "PavedDrive", "Fence",
  "MiscFeature", "MoSold", "SaleType", "SaleCondition"
)

# Natuarally ordered ordinals, some can be left as numeric, in fact
ordinals <- c("OverallQual", "OverallCond", "BsmtFullBath", "BsmtHalfBath", 
              "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
              "Fireplaces", "GarageCars")
# quality ordinals
qa_ordinals <- c(
  "ExterQual", "ExterCond", "BsmtCond", "BsmtQual", "HeatingQC", "KitchenQual", 
  "FireplaceQu", "GarageCond", "GarageQual", "PoolQC"
)

# function to properly order quality qualification ordinals
quality_ordinal <- function(x) ordered(x, levels=c("Ex","Gd", "TA", "Fa", "Po")) 
# do all data conversions
d[categorical] <- lapply(d[categorical], factor)
d[ordinals] <- lapply(d[ordinals], ordered)
d[qa_ordinals] <- lapply(d[qa_ordinals], quality_ordinal)
summary(d)

#  The advantages of xgboost in this case:
# it handles missing values
# the code is fast, multithread
# interactions and non-linearities are handled

# Take all input variables as model inputs
d_formula <- SalePrice~0+. 
# Build model matrix, allow NAs, skip the Id column
d_matrix <- model.matrix(d_formula,  model.frame(d_formula,d[,-1], na.action = na.pass))
cat("the original data size", dim(d), "\n")
cat("the design matrix size", dim(d_matrix), "\n")

# Add log to the data set (just for convinience)
d$SalePriceLog <- log(d$SalePrice)

foreach(md=c(1:6,9,12),.combine = rbind) %do% {
  params = list(objective="reg:linear", max_depth=md, eta=0.1)
  fit_cv <- xgb.cv(params=params, nfold=5, data = d_matrix[train_data_set,], 
                   nrounds = 500, label = d$SalePriceLog[train_data_set], verbose=0)
  mm <- which.min(fit_cv$evaluation_log$test_rmse_mean)
  c(max_depth=md, best_iter=mm, 
    test_error=fit_cv$evaluation_log$test_rmse_mean[mm],
    test_std=fit_cv$evaluation_log$test_rmse_std[mm])
} -> xgb_cv_res

xgb_cv_res

best_tuning_row <- which.min(xgb_cv_res[,"test_error"])
n_trees = xgb_cv_res[best_tuning_row, "best_iter"]
tree_depth = xgb_cv_res[best_tuning_row, "max_depth"]
cat("choosing trees: ", n_trees, "choosing depth:", tree_depth, "\n")

foreach(i=1:50, .combine=c) %do% {
  train_set <- sample(train_data_set, length(train_data_set)*.9)
  test_set <- setdiff(train_data_set, train_set)
  params = list(objective="reg:linear", max_depth=tree_depth, eta=0.1)
  xgb_fit <- xgboost(data=d_matrix[train_set,], label=d$SalePriceLog[train_set], 
                     params=params, nrounds = n_trees, verbose=0)
  xgb_pred <- predict(xgb_fit, newdata=d_matrix[test_set,])
  sqrt(mean((d$SalePriceLog[test_set]-xgb_pred)^2))
} -> single_model_cv_rmse

summary(single_model_cv_rmse)
options(repr.plot.width=4, repr.plot.height=3)
plot(density(single_model_cv_rmse))
