# Group 10: 
# Xinyi Ren(s2506973)
# Xinhao Liu(s2495872)
# Yingjie Yang(s2520758)

# Link of github repository:
# https://github.com/Ren0514/Practical-4-Group10.git


# Contributions to the Project
# Xinyi Ren: 35%, mainly work on backward, train function and overview comment
# Xinhao Liu: 35%, mainly work on netup, forward and train function
# Yingjie Yang: 30%, mainly work on dividing dataset and prediction
# Everyone proofread all sections as well as suggested possible optimization 
# and comments


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

# netup: initialize a list representing the network
#
# Parameters:
#   - d: a vector giving the number of nodes in each layer of a network
# 
# Returns:
#   - nn: the initialized network list with nodes, weights and offsets
netup <- function(d){
  
  # Initialize a list of nodes of each layer
  h <- lapply(d, numeric)
  
  # Initialize a list of weight matrices with elements from U(0, 0.2) random deviates
  W <- lapply(seq_along(d)[-1], function(i) {
    matrix(runif(d[i] * d[i - 1], min = 0, max = 0.2), nrow = d[i], ncol = d[i - 1])
  })
  
  # Initialize a list of offset vectors with elements from U(0, 0.2) random deviates
  b <- lapply(d[-1], function(n) runif(n, min = 0, max = 0.2))
  
  # Return a list containing initialized nodes, weight matrices, and bias vectors
  return(list(h = h, W = W, b = b))
}

# forward: compute the remaining node values implied by inp, and return the 
# updated network list
#
# Parameters:
#   - nn: a network list as built by netup
#   - inp: a vector of input values for the first layer
# 
# Returns:
#   - nn: the network list updated by forward
forward <- function(nn,inp){
  
  # Set the inp as the first layer of nodes
  nn$h[[1]] <- inp

  # iterate through hidden layers starting from the second layer
  for (l in 2:length(nn$h)) {

    # Compute the nodes in the current layer using the weights,
    # nodes of previous layer, and offsets
    nodes <- nn$W[[l-1]] %*% nn$h[[l-1]] + nn$b[[l-1]]

    # Apply the ReLU transform
    nn$h[[l]] <- pmax(nodes, 0)
  }
  
  # Return the network list after forward propagation
  return(nn)
}


# loss_derivate: Compute the derivative of the loss for class k w.r.t. the nodes 
# of output layer
#
# Parameters:
#   - last_layer: a vector of nodes in last layer
#   - k: true class label for the current input
# 
# Returns:
#   - nn: the the derivative of the loss in output layer
loss_derivative <- function(last_layer, k) {
  
  # Compute the derivative of the loss for class k
  loss_h = exp(last_layer) / sum(exp(last_layer))
  loss_h[k] = loss_h[k] - 1
  
  # Return the derivative of the loss
  return(loss_h)
}



# Backward: Compute the derivatives of L_i w.r.t. all the other nodes by working 
# backwards through the layers applying the chain rule (back-propagation)
# 
# Parameters:
#   - nn: network list returned from forward function, containing nodes, weights, 
#     and offset for each layer
#   - k: true class label for the current input, used in the computation of the 
#     last layer's derivative.
# 
# Returns:
#   - nn: the updated network list after forward
backward <- function(nn, k) {
  
  # Get the total number of layers
  total_l <- length(nn$h)
  
  # Compute the derivative of the loss w.r.t. the output of the last layer
  nn$dh[[total_l]] <- loss_derivative(nn$h[[total_l]], k)
  
  # iterate through the hidden layers in reverse order for backpropagation
  for (l in (total_l-1):1){
    
    # Compute the derivative of the loss w.r.t the nodes of the current layer
    d <- nn$dh[[l + 1]]
    
    # set the values where the node value was less than or equal to 0 to 0
    d[nn$h[[l + 1]] <= 0] <- 0
    
    # Compute the derivative of the loss w.r.t the nodes of the current layer
    nn$dh[[l]] <- t(nn$W[[l]]) %*% d
    
    # Store the derivative of the loss w.r.t the offset for the current layer
    nn$db[[l]] <- d
    
    # Compute the derivative of the loss w.r.t the weights of the current layer
    nn$dW[[l]] <- d %*% t(nn$h[[l]])

  }
  
  # Return the updated network list after backward propagation
  return(nn)

}

# train: Train a neural network by using stochastic gradient descent (SGD)
# 
# Parameters:
#   - nn: an initialized network list created by netup
#   - inp: a matrix of input data where each row corresponds to a data point
#   - k: a vector containing corresponding labels for the input data
#   - eta: step size of updating weights and offsets
#   - mb: the number of data to randomly sample to compute the gradient
#   - nstep: the number of optimization steps
# 
# Returns:
#   - nn: trained neural network
train <- function(nn,inp,k,eta=.01,mb=10,nstep=10000){
  
  # Iterate over training steps
  for (step in 1:nstep) {
    
    # generate a random sample set of length mb
    index <- sample(1:nrow(inp), mb)
    
    # Input data of random samples
    input_rows <- inp[index, ]
    
    # Corresponding labels of the samples set
    labels <- k[index]
    
    # Loop over samples set
    for (i in 1:mb) {
      
      # Get each input row
      inp_row = input_rows[i, ]
      
      # update network list by forward
      nn <- forward(nn, inp_row)
      
      # update network list by backward
      nn <- backward(nn, labels[i])
      
      # Accumulate gradients for each input row
      if (i == 1) {
        db_sum <- nn$db
        dW_sum <- nn$dW
      } else {
        db_sum <- lapply(seq_along(db_sum), function(j) db_sum[[j]] + nn$db[[j]])
        dW_sum <- lapply(seq_along(dW_sum), function(j) dW_sum[[j]] + nn$dW[[j]])
      }
      
    }
    
    # compute the average derivative of the loss based on mb samples set
    db_avg <- lapply(db_sum, function(x) x / mb)
    dW_avg <- lapply(dW_sum, function(x) x / mb)
    
    # Update weights and offsets using the averaged derivative
    nn$W <- lapply(seq_along(nn$W), function(l) nn$W[[l]] - eta * dW_avg[[l]])
    nn$b <- lapply(seq_along(nn$b), function(l) nn$b[[l]] - eta * db_avg[[l]])

  }
  
  # Return the trained neural network
  return(nn)
}


# Initialize a 4-8-7-3 network
set.seed(102)
nn <- netup(c(4, 8, 7, 3))

# Train the network
nn <- train(nn, X_train, y_train)


# predict_nn: make predictions using a trained network model by train function
#
# Parameters:
#   - nn: a trained 4-8-7-3 network
#   - test_data: a matrix of test data where each row corresponds to a data point
#
# Returns:
#   - predictions: a vector of predicted class labels for each row in the test data
predict_nn <- function(nn, test_data) {
  
  # create a predictions vector to store all predicted labels
    predictions <- sapply(1:nrow(test_data), function(i) {
    
    # use forward function get the output of each test row
    nn <- forward(nn, test_data[i, ])
    
    # get the index of the node with the maximum value at the output layer
    return(which.max(nn$h[[length(nn$h)]]))
  })
  
  # return the vector of predicted class labels
  return(predictions)
}

# Make predictions on the test set
predictions <- predict_nn(nn, X_test)

# Compute misclassification rate by comparing predictions to the true labels
misclassification_rate <- sum(predictions != y_test) / length(y_test)
cat('The misclassification rate of the tranined model is ' ,misclassification_rate , '\n')