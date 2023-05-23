# 함수형 1:1 모델
# 1. 데이터
import numpy as np

x_train = np.array(
    [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]
)
y_train = np.array(
    [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]
)

x_test = np.array([8, 9, 10])
y_test = np.array([8, 9, 10])

x_predict = np.array([11, 12, 13])

# 2 모델 구성 - 함수형 구조
from keras.models import Model
from keras.layers import Dense, Input

# 다음 레이어의 인풋이 현재 레이어의 아웃풋인 점 주의하기
input1 = Input(shape=(1,))
# input1이 dense1의 후미에 붙어 아웃풋이 됨
dense1 = Dense(100, activation="relu")(input1)  # 히든 레이어 100개
dense2 = Dense(30)(dense1)  # 히든 레이어 30개
dense3 = Dense(5)(dense2)  # 히든 레이어 5개
output1 = Dense(1)(dense3)  # 최종 아웃풋 레이어는 1개

model = Model(inputs=input1, outputs=output1)

# 3. 데이터 컴파일, 훈련하기
model.compile(loss="mse", optimizer="adam", metrics=["mse"])
model.fit(x_train, y_train, epochs=100, batch_size=1)

# 4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("mse : ", mse)

y_predict = model.predict(x_predict)
print("예측값 : \n", y_predict)
