# 101에서 110까지 구하는 코드 - 회귀 모델 사용하기
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# (1) 데이터 준비
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_test = np.array([101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
y_test = np.array([101, 102, 103, 104, 105, 106, 107, 108, 109, 110])

# (2) 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1, activation="relu"))  # 1개의 입력으로 5개의 노드 출력하기
model.add(Dense(3))  # 5개의 입력을 다시 3개의 노드로 출력하기
model.add(Dense(1, activation="relu"))  # 3개의 입력을 받아 1개로 출력하기

model.summary()  # 모델 구성 확인하기

# (3) 컴파일, 훈련
model.compile(
    loss="mse", optimizer="adam", metrics=["mse"]
)  # 기존 코드에서 mse로 변경 -> 머신이 훈련할 때 보여주는 부분이 acc대신 mse로 표현하겠다는 뜻

# validation_data에 원래 훈련과는 다른 평가용 데이터 입력하기
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_test, y_test))
loss, acc = model.evaluate(x_test, y_test, batch_size=1)

# (4) 평가, 예측
print("loss : ", loss)
print("acc : ", acc)

# 다른 데이터의 예측값 사용하기
output = model.predict(x_test)
print("결과물 : \n", output)
