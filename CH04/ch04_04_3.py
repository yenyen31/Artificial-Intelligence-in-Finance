# model=Sequential()을 사용하지 않고 함수형 모델을 이용해 모델을 만드는 코드
# 다:1 컬럼 연결 (다:다 모델에서 아웃풋만 1개인 경우) -> 인풋과 아웃풋의 reshape만 조절해주면 됨
import numpy as np

# (1) 데이터
x = np.array([range(100), range(301, 401)])
y = np.array(range(201, 301))

# np의 행과 열을 바꾸는 transpose() 이용하기
x = np.transpose(x)
# y = np.transpose(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, test_size=0.4, shuffle=False
)
x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, random_state=66, test_size=0.5, shuffle=False
)


# (2) 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()


model.add(Dense(5, input_shape=(2,), activation="relu"))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(2))

# (3) 컴파일, 훈련
model.compile(loss="mse", optimizer="adam", metrics=["mse"])
model.fit(x_train, y_train, epochs=300, batch_size=1, validation_data=(x_val, y_val))


# (4) 평가, 예측
mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)

y_predict = model.predict(x_test)
# y_predict = model.predict(x_predict)
# print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))


print("RMSE: ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score

r2_y_predict = r2.score(y_test, y_predict)
print("R2: ", r2_y_predict)
