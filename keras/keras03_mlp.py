import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([[1,2,3,4,5,6,7,8,9,10],[1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3]])
y  = np.array([11,12,13,14,15,16,17,18,19,20])

print(x.shape)
x = np.transpose(x)
print(x.shape)
print(y.shape)

x_pred = np.array([[10,1.3]])

model = Sequential()
model.add(Dense(1,input_dim=2))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=999, batch_size=1)

loss = model.evaluate(x,y)
print('loss :',loss)

result = model.predict([x_pred])
print('x의 예측값 : ',result)
