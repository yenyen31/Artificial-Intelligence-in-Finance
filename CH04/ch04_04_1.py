# model=Sequential()을 사용하지 않고 함수형 모델을 이용해 모델을 만드는 코드
# 1:1 컬럼 연결 (1개의 컬럼이 입력되는 경우)
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
# model = Sequential()

input1 = Input(shape(1, )) # Input레이어를 구성하고 입력 shape를 구성한다. 1개의 컬럼이 들어가므로 다음과 같이 설정해줌
dense1 = Dense(5, inactivation='relu')(input1) # 다음 레이어들부터는 순차형의 시퀀스형처럼 구성을 하되, 상위층에서 출력된 레이어의 이름을 하위층의 가장 끝부분에 명시해줌
dense2 = Dense(3)(dense1) # dense1과 동일하게 연결
dense3 = Dense(4)(dense2) # 동일하게 연결
output1 = Dense(1)(dense3) # 동일하게 연결

model = Model(inputs = input1, outputs=output1) # 마지막으로 Model로 전체 레이어를 엮어줌
model.summary()

# (3) 컴파일, 훈련
model.compile(loss="mse", optimizer="adam", metrics=["mse"])
model.fit(x_train, y_train, epochs=300, batch_size=1, validation_data=(x_val, y_val))


# (4) 평가, 예측
mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)

y_predict = model.predict(x_test)
# y_predict = model.predict(x_predict)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))


print("RMSE: ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score

r2_y_predict = r2.score(y_test, y_predict)
print("R2: ", r2_y_predict)
