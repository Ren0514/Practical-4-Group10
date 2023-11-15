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

train <- function(nn,inp,k,eta=.01,mb=10,nstep=1){
  
  for (step in 1:nstep) {
    
    index <- sample(1:nrow(inp), mb)# generate a random sample set
    
    input_rows <- inp[index, ]# Input data of random samples
    
    labels <- k[index]# Corresponding labels
    
    for (i in 1:mb) {# Loop over samples set
      
      inp_row = as.vector(input_rows[i, ])
      
      nn <- forward(nn, inp_row)
      
      nn <- backward(nn, labels[i])
      
    
    for (l in 1:length(nn$W)){
      nn$W[[l]] <- nn$W[[l]] - eta*nn$dW[[l]]
      nn$b[[l]] <- nn$b[[l]] - eta*nn$db[[l]]
    }

  }
  
  return(nn)
  }
}