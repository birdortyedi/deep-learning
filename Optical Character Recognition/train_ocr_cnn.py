import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten,SpatialDropout2D
from tensorflow.contrib.keras.python.keras.utils import plot_model
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping,ReduceLROnPlateau

print("Loading data...")

collection = np.load('collection.npy')
labels_onehot = np.load('labels_onehot.npy')
collection,labels_onehot = shuffle(collection,labels_onehot)

x_train_full,x_valid,y_train_full,y_valid = train_test_split(collection,labels_onehot,test_size=0.2)
x_train,x_test,y_train,y_test = train_test_split(x_train_full,y_train_full,test_size=0.25)

np.save('x_test.npy',x_test)
np.save('y_test.npy',y_test)

print ("Making model...")

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(20, 20, 1), data_format='channels_last', padding='same',activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(SpatialDropout2D(0.25,data_format='channels_last'))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same',activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(SpatialDropout2D(0.25,data_format='channels_last'))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same',activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(SpatialDropout2D(0.25,data_format='channels_last'))

model.add(Flatten())

model.add(Dense(units=32,activation='sigmoid'))
model.add(Dense(units=64,activation='sigmoid'))
model.add(Dense(units=26, activation='softmax'))

print ("Compiling model...")

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# print("Visualizing model...")
# model.summary()
# plot_model(model,show_shapes=True, to_file='model_visualized.png')

train_datagen = ImageDataGenerator(rotation_range=10.,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.1,zoom_range=0.1)
valid_datagen = ImageDataGenerator(rotation_range=10.,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.1,zoom_range=0.1)

train_datagen.fit(x_train)
valid_datagen.fit(x_valid)

tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=False, write_images=False)
mcCallBack = ModelCheckpoint(filepath='cnn_best_model_2.h5', verbose=1, save_best_only=True)
esCallBack = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=20, verbose=1, mode='auto')
rlrPlateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

print ("Starting to fit the model...")

model.fit_generator(generator = train_datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32,
                    verbose=1,
                    validation_data = valid_datagen.flow(x_valid, y_valid, batch_size=32),
                    validation_steps=len(x_valid)/32,
                    workers = 4,
                    epochs = 500,
                    callbacks=[tbCallBack,mcCallBack,esCallBack,rlrPlateau])
