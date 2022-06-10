Basic Deep Learning and Neutral Network

###Introduction to Deep Learning###记模型图比较好用
#interations:neutral network account for interactions really well.Text, image, video, audio...
#Build and tune deep learning models using keras
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
predictors = np.loadtxt('predictors_data.csv', delimiter=',')
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

#Forword Propagation#
#Basic Idea: multiply-add process
import numpy as np
input_data = np.array([2, 3])
weights = {'node_0': np.array([1, 1]),
            'node_1': np.array([-1, 1]),
            'output': np.array([2, -1])}
node_0_value = (input_data * weights['node_0']).sum()
node_1_value = (input_data * weights['node_1']).sum()

hidden_layer_values = np.array([node_0_value, node_1_value])
print(hidden_layer_values)
#return:[5, 1]
output = (hidden_layer_values * weights['output']).sum()
print(output)
#return:9
 
##Action Functions##
#allows the models to capture non-linearities
#ReLU:Rectified Linear Activation. if x<0->ReLU(x)=0;if x>=0->ReLU(x)=x
#tanh:双曲正切，hyperbolic tangent function. tanh x=sinh x/cosh x=(e^x-e^-x)/(e^x+e^-x)∈[-1,1]
def relu(input):
    '''Define your relu activation function here'''
    # Calculate the value for the output of the relu function: output
    output = max(0, input)
    # Return the value just calculated
    return(output)

#apply relu function or tanh:
import numpy as np
input_data = np.array([-1, 2])
weights = {'node_0': np.array([3, 3]),
               'node_1': np.array([1, 5]),
               'output': np.array([2, -1])}
# Calculate node 0 value: node_0_output
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = np.tanh/relu(node_0_input) 
# Calculate node 1 value: node_1_output
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = np.tanh/relu(node_1_input) #we apply tanh function to convert input to output
# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_output, node_1_output])
# Calculate model output (do not apply relu)
output = (hidden_layer_output * weights['output']).sum()
print(output)
#return：1.2382242525694254

##Deeper Network##->multiple hidden layers
#It is a kind of representation learning, partially replace the need of feature engineering
#EG:do forward propagation for a neural network with 2 hidden layers.
def predict_with_network(input_data):
    # Calculate node 0 in the first hidden layer
    node_0_0_input = (input_data * weights['node_0_0']).sum()
    node_0_0_output = relu(node_0_0_input)

    # Calculate node 1 in the first hidden layer
    node_0_1_input = (input_data * weights['node_0_1']).sum()
    node_0_1_output = relu(node_0_1_input)

    # Put node values into array: hidden_0_outputs
    hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])

    # Calculate node 0 in the second hidden layer
    node_1_0_input = (hidden_0_outputs * weights['node_1_0']).sum()
    node_1_0_output = relu(node_1_0_input)

    # Calculate node 1 in the second hidden layer
    node_1_1_input = (hidden_0_outputs * weights['node_1_1']).sum()
    node_1_1_output = relu(node_1_1_input)

    # Put node values into array: hidden_1_outputs
    hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])
    
    # Calculate output here: model_output
    model_output = (hidden_1_outputs * weights['output']).sum()
    
    # Return model_output
    return(model_output)
output = predict_with_network(input_data)
print(output)
#return:182

