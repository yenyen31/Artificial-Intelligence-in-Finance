# Sequential 1:1 모델
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

x_predict = np.array([11, 12, 13]) # 트러블 슈팅 *** 책과 다르게 수정한 부분 ***

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

# Sequential 모델 사용하기
model = Sequential()


model.add(Dense(100, input_dim=1, activation="relu"))  # 첫 번째 히든 레이어는 100개
model.add(Dense(30))  # 두 번째 히든 레이어는 30개
model.add(Dense(5))  # 세 번째 히든 레이어는 5개
model.add(Dense(1))  # 네 번째 히든 레이어는 1개

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
