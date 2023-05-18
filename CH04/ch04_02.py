# 데이터 분리해서 사용하기
import numpy as np

# (1) 데이터
x = np.array(range(1, 101))
y = np.array(range(1, 101))
x_train = x[:60]
x_val = x[60:80]
x_test = x[80:]
y_train = y[:60]
y_val = x[60:80]
y_test = x[80:]

x_predict = np.array(range(101, 111))


# (2) 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

# model.add(Dense(5, input_dim=1, activation="relu"))  # 1개의 입력으로 5개의 노드 출력하기
model.add(Dense(5, input_shape=(1,), activation="relu"))
model.add(Dense(3))  # 5개의 입력을 다시 3개의 노드로 출력하기
model.add(Dense(4))
model.add(Dense(1))

# model.summary()  # 모델 구성 확인하기

# (3) 컴파일, 훈련
model.compile(loss="mse", optimizer="adam", metrics=["mse"])
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_data=(x_val, y_val))


# (4) 평가, 예측
mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)

# y_predict = model.predict(x_test)
y_predict = model.predict(x_predict)
print(y_predict)
