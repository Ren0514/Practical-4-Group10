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

## netup: initialize a list representing the network
## d: a vector giving the number of nodes in each layer of a network. 
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

## forward: compute the remaining node values implied by inp, and return the 
## updated network list
## nn: a network list as built by netup
## inp: a vector of input values for the first layer
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

## loss_derivate: Compute the derivative of the loss for class k 
## w.r.t. the nodes of output layer
## last_layer: a vector of nodes in last layer
## k: true class label for the current input
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

