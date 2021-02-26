
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input, BatchNormalization, MaxPool2D
from keras.utils import to_categorical

# This function runs a new model
def run_new_model(x_train, x_test, train_lable, test_lable, Batchnorm = False, Droprate = 0.5, Batchsize = None, epochs = 20):
    
    test_lable = to_categorical(test_lable)         #prepare data
    train_lable = to_categorical(train_lable)

    model = Sequential()
    model.add(Conv2D(filters=96, kernel_size=(6,6), input_shape = (32, 32, 3), strides=(2,2), activation='relu'))
    if Batchnorm == True:
        model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2), strides = (1,1)))
    model.add(Conv2D(filters=96, kernel_size=(3,3), activation='relu', padding="same"))
    if Batchnorm == True:
        model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2), strides = (1,1)))
    model.add(Conv2D(filters=96, kernel_size = (3,3), activation='relu', padding="same"))
    if Batchnorm == True:
        model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2), strides = (1,1)))
    model.add(Flatten())
    model.add(Dropout(Droprate))
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dropout(Droprate))
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dropout(Droprate))
    model.add(Dense(10, activation = 'softmax'))

    model.summary()	    #summary of the fitted model
    model.compile(                                 #define model parameters
        optimizer = 'adam',
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )
    history = model.fit(x_train, train_lable, batch_size = Batchsize, validation_data = (x_test, test_lable), epochs = epochs)

    return model, history.history