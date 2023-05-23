# Sequential 다:다 모델
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

# reshape 하기
x_train = np.transpose(x_train)
y_train = np.transpose(y_train)
x_test = np.transpose(x_test)
y_test = np.transpose(y_test)
x_predict = np.transpose(x_predict)


# reshape 확인하기
print(x_train.shape)
print(x_test.shape)
print(x_predict.shape)

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(100, input_dim=2, activation="relu"))
model.add(Dense(30))
model.add(Dense(5))
model.add(Dense(1))

# 3. 훈련하기
model.compile(
    loss="mse", optimizer="adam", metrics=["mse"]
)  # loss는 mse로 설정해 최저 손실 값 구하기
model.fit(x_train, y_train, epochs=100, batch_size=1)  # 100번 훈련시키고, 배치 사이즈는 1개씩 잘라 사용하기

# 4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)  # x_test, y_test로 훈련시키고
print("mse : ", mse)

y_predict = model.predict(x_predict)  # x_predict 11, 12, 13으로 예측값 확인하기
print("예측값 : \n", y_predict)
