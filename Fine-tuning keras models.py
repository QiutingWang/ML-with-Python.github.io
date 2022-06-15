Fine-tuning keras models
###Understanding Model Optimization###
#Optimization is hard: when learning rate is low,update too low.It needs to optimize 1000 parameters with complex relationship
#When learning rate is too low/high,poor choice of activation function can lead to losses in the first few epoches.

#Stochastic Gradient Desecent(SGD),to see how the different learning rates use the simplest optimizer.
def get_new_model(input_shape = input_shape):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape = input_shape))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))#用于任务分类，对于（0，1）的vector进行加权、归一化，使sum=1，考虑范围内所有值https://medium.com/artificialis/softmax-function-and-misconception-4248917e5a1c
    return(model)
#其他激活函数：relu/leaky relu/sigmoid
lr_to_test = [.000001, 0.01, 1]
# loop over learning rates
for lr in lr_to_test:
   model = get_new_model()
   my_optimizer = SGD(lr=lr)
   model.compile(optimizer = my_optimizer, loss = 'categorical_crossentropy')
   model.fit(predictors, target)
#return:
Testing model with learning rate: 0.000001
   
    Epoch 1/10
    
 32/891 [>.............................] - ETA: 43s - loss: 3.6053
448/891 [==============>...............] - ETA: 1s - loss: 3.6835 
891/891 [==============================] - 1s - loss: 3.6057     
    Epoch 2/10
    
 32/891 [>.............................] - ETA: 0s - loss: 3.5751
512/891 [================>.............] - ETA: 0s - loss: 3.5372
891/891 [==============================] - 0s - loss: 3.5656     
    Epoch 3/10
    
 32/891 [>.............................] - ETA: 0s - loss: 2.6692
448/891 [==============>...............] - ETA: 0s - loss: 3.3878
832/891 [===========================>..] - ETA: 0s - loss: 3.5528
891/891 [==============================] - 0s - loss: 3.5255     
    Epoch 4/10
    
 32/891 [>.............................] - ETA: 0s - loss: 3.0058
448/891 [==============>...............] - ETA: 0s - loss: 3.4965
891/891 [==============================] - 0s - loss: 3.4854     
    Epoch 5/10
    
 32/891 [>.............................] - ETA: 0s - loss: 2.5452
480/891 [===============>..............] - ETA: 0s - loss: 3.4640
891/891 [==============================] - 0s - loss: 3.4454     
    Epoch 6/10
    
 32/891 [>.............................] - ETA: 0s - loss: 3.4446
480/891 [===============>..............] - ETA: 0s - loss: 3.4553
891/891 [==============================] - 0s - loss: 3.4056     
    Epoch 7/10
    
 32/891 [>.............................] - ETA: 0s - loss: 4.1073
576/891 [==================>...........] - ETA: 0s - loss: 3.4968
891/891 [==============================] - 0s - loss: 3.3659     
    Epoch 8/10
    
 32/891 [>.............................] - ETA: 0s - loss: 3.0972
512/891 [================>.............] - ETA: 0s - loss: 3.1786
891/891 [==============================] - 0s - loss: 3.3263     
    Epoch 9/10
    
 32/891 [>.............................] - ETA: 0s - loss: 3.7464
416/891 [=============>................] - ETA: 0s - loss: 3.1884
832/891 [===========================>..] - ETA: 0s - loss: 3.2820
891/891 [==============================] - 0s - loss: 3.2867     
    Epoch 10/10
    
 32/891 [>.............................] - ETA: 0s - loss: 3.3862
448/891 [==============>...............] - ETA: 0s - loss: 3.1540
864/891 [============================>.] - ETA: 0s - loss: 3.2075
891/891 [==============================] - 0s - loss: 3.2473     
    
    
    Testing model with learning rate: 0.010000
    
    Epoch 1/10
    
 32/891 [>.............................] - ETA: 38s - loss: 1.0910
480/891 [===============>..............] - ETA: 1s - loss: 2.0211 
891/891 [==============================] - 1s - loss: 1.4185     
    Epoch 2/10
    
 32/891 [>.............................] - ETA: 0s - loss: 2.0775
512/891 [================>.............] - ETA: 0s - loss: 0.7657
891/891 [==============================] - 0s - loss: 0.7036     
    Epoch 3/10
    
 32/891 [>.............................] - ETA: 0s - loss: 0.5699
448/891 [==============>...............] - ETA: 0s - loss: 0.7054
891/891 [==============================] - 0s - loss: 0.6473     
    Epoch 4/10
    
 32/891 [>.............................] - ETA: 0s - loss: 0.6218
384/891 [===========>..................] - ETA: 0s - loss: 0.6580
800/891 [=========================>....] - ETA: 0s - loss: 0.6330
891/891 [==============================] - 0s - loss: 0.6244     
    Epoch 5/10
    
 32/891 [>.............................] - ETA: 0s - loss: 0.4922
384/891 [===========>..................] - ETA: 0s - loss: 0.6059
768/891 [========================>.....] - ETA: 0s - loss: 0.6165
891/891 [==============================] - 0s - loss: 0.6236     
    Epoch 6/10
    
 32/891 [>.............................] - ETA: 0s - loss: 0.6751
448/891 [==============>...............] - ETA: 0s - loss: 0.6108
800/891 [=========================>....] - ETA: 0s - loss: 0.6059
891/891 [==============================] - 0s - loss: 0.6014     
    Epoch 7/10
    
 32/891 [>.............................] - ETA: 0s - loss: 0.6260
416/891 [=============>................] - ETA: 0s - loss: 0.6095
832/891 [===========================>..] - ETA: 0s - loss: 0.6030
891/891 [==============================] - 0s - loss: 0.6029     
    Epoch 8/10
    
 32/891 [>.............................] - ETA: 0s - loss: 0.6284
384/891 [===========>..................] - ETA: 0s - loss: 0.5976
736/891 [=======================>......] - ETA: 0s - loss: 0.6039
891/891 [==============================] - 0s - loss: 0.6136     
    Epoch 9/10
    
 32/891 [>.............................] - ETA: 0s - loss: 0.6793
448/891 [==============>...............] - ETA: 0s - loss: 0.6105
832/891 [===========================>..] - ETA: 0s - loss: 0.6008
891/891 [==============================] - 0s - loss: 0.6059     
    Epoch 10/10
    
 32/891 [>.............................] - ETA: 0s - loss: 0.6410
448/891 [==============>...............] - ETA: 0s - loss: 0.5846
832/891 [===========================>..] - ETA: 0s - loss: 0.5890
891/891 [==============================] - 0s - loss: 0.5901     
    
    
    Testing model with learning rate: 1.000000
    
    Epoch 1/10
    
 32/891 [>.............................] - ETA: 36s - loss: 1.0273
512/891 [================>.............] - ETA: 1s - loss: 5.4474 
891/891 [==============================] - 1s - loss: 5.9885     
    Epoch 2/10
    
 32/891 [>.............................] - ETA: 0s - loss: 4.5332
512/891 [================>.............] - ETA: 0s - loss: 6.0128
832/891 [===========================>..] - ETA: 0s - loss: 6.1024
891/891 [==============================] - 0s - loss: 6.1867     
    Epoch 3/10
    
 32/891 [>.............................] - ETA: 0s - loss: 7.0517
416/891 [=============>................] - ETA: 0s - loss: 5.9280
704/891 [======================>.......] - ETA: 0s - loss: 6.2961
891/891 [==============================] - 0s - loss: 6.1867     
    Epoch 4/10
    
 32/891 [>.............................] - ETA: 0s - loss: 6.0443
416/891 [=============>................] - ETA: 0s - loss: 6.3155
864/891 [============================>.] - ETA: 0s - loss: 6.1376
891/891 [==============================] - 0s - loss: 6.1867     
    Epoch 5/10
    
 32/891 [>.............................] - ETA: 0s - loss: 9.0664
416/891 [=============>................] - ETA: 0s - loss: 6.2768
864/891 [============================>.] - ETA: 0s - loss: 6.1562
891/891 [==============================] - 0s - loss: 6.1867     
    Epoch 6/10
    
 32/891 [>.............................] - ETA: 0s - loss: 6.0443
416/891 [=============>................] - ETA: 0s - loss: 6.2380
864/891 [============================>.] - ETA: 0s - loss: 6.1749
891/891 [==============================] - 0s - loss: 6.1867     
    Epoch 7/10
    
 32/891 [>.............................] - ETA: 0s - loss: 5.0369
512/891 [================>.............] - ETA: 0s - loss: 6.4850
891/891 [==============================] - 0s - loss: 6.1867     
    Epoch 8/10
    
 32/891 [>.............................] - ETA: 0s - loss: 5.0369
480/891 [===============>..............] - ETA: 0s - loss: 6.0107
891/891 [==============================] - 0s - loss: 6.1867     
    Epoch 9/10
    
 32/891 [>.............................] - ETA: 0s - loss: 5.5406
384/891 [===========>..................] - ETA: 0s - loss: 6.0863
832/891 [===========================>..] - ETA: 0s - loss: 6.1024
891/891 [==============================] - 0s - loss: 6.1867     
    Epoch 10/10
    
 32/891 [>.............................] - ETA: 0s - loss: 5.5406
480/891 [===============>..............] - ETA: 0s - loss: 6.5480
891/891 [==============================] - 0s - loss: 6.1867     
#Dying Neuron Problem#
#the problem occurs when a neuron< 0, for all rows of your data. EG: LeRU activation
#Vanishing gradients:occurs when many layers have small slopes

###Model Validation###your model perform on the training data is not a good indication of how it will perform on the new data.
#Vaildation data is explicity held out from training, and used only to test model performance
#Validation in DL#
#few people use k-fold cross validation on deep learning-> deep learning almostly uses large datasets
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy']) 
model.fit(predictors, target, validation_split=0.3)
#return:
Epoch 1/10
89648/89648 [=====] - 3s - loss: 0.7552 - acc: 0.5775 - val_loss: 0.6969 - val_acc: 0.5561
Epoch 2/10
89648/89648 [=====] - 4s - loss: 0.6670 - acc: 0.6004 - val_loss: 0.6580 - val_acc: 0.6102
...
Epoch 8/10
89648/89648 [=====] - 5s - loss: 0.6578 - acc: 0.6125 - val_loss: 0.6594 - val_acc: 0.6037
Epoch 9/10
89648/89648 [=====] - 5s - loss: 0.6564 - acc: 0.6147 - val_loss: 0.6568 - val_acc: 0.6110
Epoch 10/10
89648/89648 [=====] - 5s - loss: 0.6555 - acc: 0.6158 - val_loss: 0.6557 - val_acc: 0.6126
                
#early stopping: stop training when the valiation score isn't improving
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=2) #the model can go without improving before we stop training
model.fit(predictors, target, validation_split=0.3, nb_epoch=20,callbacks = [early_stopping_monitor]) #nb_epoch=high maximum number of epochs
#return:
Train on 89648 samples, validate on 38421 samples
Epoch 1/20
89648/89648 [====] - 5s - loss: 0.6550 - acc: 0.6151 - val_loss: 0.6548 - val_acc: 0.6151
Epoch 2/20
89648/89648 [====] - 6s - loss: 0.6541 - acc: 0.6165 - val_loss: 0.6537 - val_acc: 0.6154
...
Epoch 8/20
89648/89648 [====] - 6s - loss: 0.6527 - acc: 0.6181 - val_loss: 0.6531 - val_acc: 0.6160
Epoch 9/20
89648/89648 [====] - 7s - loss: 0.6524 - acc: 0.6176 - val_loss: 0.6513 - val_acc: 0.6172
Epoch 10/20
89648/89648 [====] - 6s - loss: 0.6527 - acc: 0.6176 - val_loss: 0.6549 - val_acc: 0.6134
Epoch 11/20
89648/89648 [====] - 6s - loss: 0.6522 - acc: 0.6178 - val_loss: 0.6517 - val_acc: 0.6169
#Experiment:change nodes, change layers' number

##Model Capacity##
#start from a small network,
#gradually increase capacity,
#keep it until validation score is no longer improving
