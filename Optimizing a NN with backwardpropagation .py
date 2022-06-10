Optimizing a neural network with backward propagation
###The need for Optimizing###
#changing weights improve the model for the data point
#EG:
# The data point you will make a prediction for
input_data = np.array([0, 3])
# Sample weights
weights_0 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [1, 1]
            }
# The actual target value, used to calculate the error
target_actual = 3
# Make prediction using original weights
model_output_0 = predict_with_network(input_data, weights_0)
# Calculate error: error_0
error_0 = model_output_0 - target_actual
# Create weights that cause the network to make perfect prediction (3): weights_1
weights_1 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [1, 0]
            }
# Make prediction using new weights: model_output_1
model_output_1 = predict_with_network(input_data, weights_1)
# Calculate error: error_1
error_1 = model_output_1 - target_actual
# Print error_0 and error_1
print(error_0) #6
print(error_1) #0

#Loss function:Aggregating errors in predictions from many data points into single number,measure the prediction performance,lower value, better model
#Squared Error loss function:total sqared error->mean sqared error
from sklearn.metrics import mean_squared_error
# Create model_output_0 
model_output_0 = []
# Create model_output_1
model_output_1 = []
# Loop over input_data
for row in input_data:
    # Append prediction to model_output_0
    model_output_0.append(predict_with_network(row,weights_0))
    # Append prediction to model_output_1
    model_output_1.append(predict_with_network(row,weights_1))
# Calculate the mean squared error for model_output_0: mse_0
mse_0 = mean_squared_error(model_output_0,target_actuals)
# Calculate the mean squared error for model_output_1: mse_1
mse_1 = mean_squared_error(model_output_1,target_actuals)
# Print mse_0 and mse_1
print("Mean squared error with weights_0: %f" %mse_0) #Mean squared error with weights_0: 37.500000
print("Mean squared error with weights_1: %f" %mse_1) #Mean squared error with weights_1: 49.890625

#use Gradient Descent to find the minimum value of loss function
#gradient descent:start at a random point,until you are flat:find the slope,take a step downhill
#Slope>0:going opposite the slope means moving to lower numbers,substract the slope from the current value,too big one step for the next will lead to astray
# -->solution:learning rate.Update each weight by substracting "learning rate*slope". normally, learning rate is around 0.01
#Slope Caluclation:multiple together:1.slope of the loss function wrt value at the node we feed to[2*(predicted value-actrual value)]2.the value of the node that feeds into our weight3.slope of activation function value we feed into
import numpy as np
weights = np.array([1, 2])
input_data = np.array([3, 4])
target = 6
learning_rate = 0.01
preds = (weights * input_data).sum()
error = preds - target
print(error) #5
gradient = 2 * input_data * error
gradient
#array([30, 40])
weights_updated = weights - learning_rate * gradient
preds_updated = (weights_updated * input_data).sum()
error_updated = preds_updated - target
print(error_updated) #2.5

#Making multiple updates to weights
n_updates = 20
mse_hist = []
# Iterate over the number of updates
for i in range(n_updates):
    # Calculate the slope: slope
    slope = get_slope(input_data, target, weights)
    # Update the weights: weights
    weights = weights - 0.01 * slope
    # Calculate mse with new weights: mse
    mse = get_mse(input_data, target, weights)
    # Append the mse to mse_hist
    mse_hist.append(mse)
# Plot the mse history
plt.plot(mse_hist)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()

#Backpropagation:from right diagram to left
#go back one layer at a time
