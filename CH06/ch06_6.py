# 함수형 다:다 모델
# 1. 데이터

import numpy as np

# 1~7까지의 데이터와 11~17데이터로 훈련을 시킨 다음, 8,9,10과 18,19,20을 이용해 평가
# 그리고 21,22,23과 31,32,33이 잘 예측되는 지 확인하는 모델 만들기
x_train = np.array(
    [
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
        ],
        [11, 12, 13, 14, 15, 16, 17],
    ]
)
y_train = np.array(
    [
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
        ],
        [11, 12, 13, 14, 15, 16, 17],
    ]
)

x_test = np.array([[8, 9, 10], [18, 19, 20]])
y_test = np.array([[8, 9, 10], [18, 19, 20]])

x_predict = np.array([[21, 22, 32], [31, 32, 33]])

_train = np.transpose(x_train)
y_train = np.transpose(y_train)
x_test = np.transpose(x_test)
y_test = np.transpose(y_test)
x_predict = np.transpose(x_predict)


# 2 모델 구성 - 함수형 구조
from keras.models import Model
from keras.layers import Dense, Input

input1 = Input(shape=(2,))
dense1 = Dense(100, activation="relu")(input1)
dense2 = Dense(30)(dense1)
dense3 = Dense(5)(dense2)
output1 = Dense(2)(dense3)

model = Model(inputs=input1, outputs=output1)

# 3. 데이터 컴파일, 훈련하기
model.compile(loss="mse", optimizer="adam", metrics=["mse"])
model.fit(x_train, y_train, epochs=100, batch_size=1)

# 4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)

y_predict = model.predict(x_predict)
print("예측값 : \n", y_predict)
