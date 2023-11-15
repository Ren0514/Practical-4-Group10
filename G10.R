# Group 10: 
# Xinyi Ren(s2506973)
# Xinhao Liu(s2495872)
# Yingjie Yang(s2520758)

# Load the Iris dataset
data(iris)

# Transform label column for the species into numeric
iris$Species <- as.numeric(iris[,5])

# Convert the Iris dataset to a matrix
iris <- unname(data.matrix(iris))

## Divide the iris data into training data and test data
# obtain indices for the test data (every 5th row starting from row 5)
test_indices <- seq(5, nrow(iris), by = 5)

# Create the test data
test_data <- iris[test_indices, ]

# Create the training data by excluding the test data
train_data <- iris[-test_indices, ]

# Extract characteristics (X) and labels (y) for training data
X_train <- train_data[, 1:4]
y_train <- train_data[,5]

# Extract characteristics (X) and labels (y) for testing data
X_test <- test_data[, 1:4]
y_test <- test_data[,5]
