# 1에서 10까지의 훈련 모델을 만드는 코드
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # 데이터 생성
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # 결과값이 들어가는 y데이터


# 딥러닝 모델 구상
# 케라스의 순차적 모델 - 회귀 모델 이용
model = Sequential()
model.add(Dense(1, input_dim=1, activation="relu"))

model = Sequential()  # 딥러닝 모델을 순차적으로 구성함
# 순차적 구성 모델에 Dense layer을 추가
model.add(Dense(1, input_dim=1, activation="relu"))

# 실행 위해 컴파일 하기
model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])

# 실행
model.fit(x, y, epochs=500, batch_size=1)

loss, acc = model.evaluate(x, y, batch_size=1)

print("loss: : ", loss)
print("acc : ", acc)
