import numpy as np
np.random.seed(1)

# Transfer function, sigmoid is really outdated but works fine for the purpose of this demo
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Transfer function's derivitive
def sigmoid_derivative(x):
    return x * (1 - x)
    
# Creates an inputs-by-outputs matrix with random floating numbers with a mean of 0
# and a standard deviation of .5
def initialize_weights(inputs, outputs):
    return 2*np.random.random((inputs,outputs)) - 1

# Input and output data for the neural net. 3-input XOR gate truth table. See README.md
input_data = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0]])             
output_data = np.array([[0], [1], [1], [0], [1], [0], [0]])

# Parameters which dictates the architecture of the neural net
input_size = 3
hidden_layer_size = 3
output_size = 1

# Randomly initialize the weights
weights_0 = initialize_weights(input_size, hidden_layer_size)
weights_1 = initialize_weights(hidden_layer_size, output_size)

# configuration:
alpha = 1.2 #The learning rate
iterations = 100000 #Amount of iterations, aka epochs
print_every = 10000 #Print the mean error of the network every 10000 iteration
for i in range(iterations):
    # Forward Propagation
    # input data flows through layer 1 and layer 2
    layer_0 = input_data
    layer_1 = sigmoid(np.dot(layer_0, weights_0))
    layer_2 = sigmoid(np.dot(layer_1, weights_1))

    # Error calculation
    # output_data is the expected output and layer_2 contains the real output from the network
    final_error = output_data - layer_2
    
    # Backpropegation
    layer_2_delta = final_error * sigmoid_derivative(layer_2)
    layer_1_error = layer_2_delta.dot(weights_1.T) #OBS weights_1 is transposed before the dot product
    layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)

    # Update the weights
    weights_1 += alpha * layer_1.T.dot(layer_2_delta) #OBS layer_1 is transposed before the dot product
    weights_0 += alpha * layer_0.T.dot(layer_1_delta) #OBS layer_0 is transposed before the dot product

    if not i % print_every: 
        mean_error = np.mean(np.abs(final_error))
        print("Mean error:", mean_error)

print("Training complete!")

# Present new data for the neural network, for testing purposes
new_data = np.array([1, 1, 1])
layer_1 = sigmoid(np.dot(new_data, weights_0))
layer_2 = sigmoid(np.dot(layer_1, weights_1))

# Print the result
rounded_res = round(float(layer_2))
print("Result for new data:", rounded_res) #Should be 1
