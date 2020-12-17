from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

#File variable
dataset = loadtxt('diabetes.csv', delimiter=',')
#Split data into x and y inputs
X = dataset[:,0:8]
y = dataset[:,8]
#Parameterize the model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#Run and store model history into variable
history = model.fit(X, y, validation_split=0.33, epochs=150, batch_size=10)
#Print keys being used to plot
print(history.history.keys())
#Plot the results with matplotlib
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
