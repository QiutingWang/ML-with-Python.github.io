Building deep learning models with Keras
###Creating a Keras Model###
#Steps:
#1.Specify architecture:how many layers? how many nodes in each layer? what activation function do you want to use in each layer?
#2.Complie the model:specify the loss function, how optimization works, 
#3.fit the model: the optimization of model weights 
#4.Prediction
#model specification
import numpy as np #reading data
from keras.layers import Dense
from keras.models import Sequential #for building our model
predictors = np.loadtxt('predictors_data.csv', delimiter=',')
n_cols = predictors.shape[1] #find the # of nodes in the input layer<->specify # of column
model = Sequential() #sequntial model requires that each layer has weights or connections only to the one layer coming directly after it in the network diagram
model.add(Dense(100, activation='relu', input_shape = (n_cols,))) #we add laryers use add() method, Dense:standard layer type, all the nodes in the previous layer connect to all of the nodes in the current layer
model.add(Dense(100, activation='relu')) #we specify the number of nodes as the 100 #(n_cols,):it can have any number of rows
model.add(Dense(1)) #the output laryer.
#this model has 2 hiden layers and an output layer

###Compiling and fitting a model###
#specify the optimizer,which control the learning rate,how quickly your model finds good weights
#specify your loss function:"mean_squared_error" for regression problems
model.compile(optimizer='adam', loss='mean_squared_error') #more information about adam method:https://arxiv.org/pdf/1412.6980v8.pdf

###Fitting a model###
#apply backpropagation and gradient descent with your data to update the weights
#scaling the data before fitting
...
model.fit(predictors, target) # first argument:predictive features (predictors), and the data to be predicted (target) is the second argument.
#return:
    Epoch 1/10
 32/534 [>.............................] - ETA: 22s - loss: 146.0927
534/534 [==============================] - 1s - loss: 78.0060       
    Epoch 2/10
 32/534 [>.............................] - ETA: 0s - loss: 84.9695
534/534 [==============================] - 0s - loss: 30.3033     
    Epoch 3/10
 32/534 [>.............................] - ETA: 0s - loss: 21.0460
534/534 [==============================] - 0s - loss: 27.0827     
    Epoch 4/10
 32/534 [>.............................] - ETA: 0s - loss: 16.8378
534/534 [==============================] - 0s - loss: 25.1230     
    Epoch 5/10
 32/534 [>.............................] - ETA: 0s - loss: 23.2098
480/534 [=========================>....] - ETA: 0s - loss: 21.9340
534/534 [==============================] - 0s - loss: 24.0225     
    Epoch 6/10
 32/534 [>.............................] - ETA: 0s - loss: 13.3955
448/534 [========================>.....] - ETA: 0s - loss: 23.5183
534/534 [==============================] - 0s - loss: 23.2020     
    Epoch 7/10
 32/534 [>.............................] - ETA: 0s - loss: 28.1708
512/534 [===========================>..] - ETA: 0s - loss: 22.5534
534/534 [==============================] - 0s - loss: 22.4516     
    Epoch 8/10
 32/534 [>.............................] - ETA: 0s - loss: 11.3862
534/534 [==============================] - 0s - loss: 22.0787     
    Epoch 9/10
 32/534 [>.............................] - ETA: 0s - loss: 21.9371
534/534 [==============================] - 0s - loss: 21.7463     
    Epoch 10/10
 32/534 [>.............................] - ETA: 0s - loss: 5.4711
534/534 [==============================] - 0s - loss: 21.5530    

###Classification Model###
#set  'categorical_crossentropy' as most common loss function.LogLoss, the lower the better
#metrics = ['accuracy']
#modify the last layer,output,use 'softmax' as activation function.it has a separate node for each potential outcome.
from keras.utils.np_utils import to_categorical 
data = pd.read_csv('basketball_shot_log.csv')
predictors = data.drop(['shot_result'], axis=1).as_matrix() #drop out un-target columns,and store them into a numpy metrix
target = to_categorical(data.shot_result) #import utility function to convert data from one column to multiple columns
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(predictors, target)
Epoch 1/10
128069/128069 [==============================] - 4s - loss: 0.7706 - acc: 0.5759
Epoch 2/10
128069/128069 [==============================] - 5s - loss: 0.6656 - acc: 0.6003
Epoch 3/10
128069/128069 [==============================] - 6s - loss: 0.6611 - acc: 0.6094
Epoch 4/10
128069/128069 [==============================] - 7s - loss: 0.6584 - acc: 0.6106
Epoch 5/10
128069/128069 [==============================] - 7s - loss: 0.6561 - acc: 0.6150
Epoch 6/10
128069/128069 [==============================] - 9s - loss: 0.6553 - acc: 0.6158
Epoch 7/10
128069/128069 [==============================] - 9s - loss: 0.6543 - acc: 0.6162
Epoch 8/10
128069/128069 [==============================] - 9s - loss: 0.6538 - acc: 0.6158
Epoch 9/10
128069/128069 [==============================] - 10s - loss: 0.6535 - acc: 0.6157
Epoch 10/10
128069/128069 [==============================] - 10s - loss: 0.6531 - acc: 0.6166

###Saving,reload and use model###
from keras.models import load_model
model.save('model_file.h5')
my_model = load_model('my_model.h5')
predictions = my_model.predict(data_to_predict_with)
probability_true = predictions[:,1] #find the column corresponding to predicted probabilities of survival being True.
my_model.summary()
