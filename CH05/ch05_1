# ch05
# 1. concatenate

# 1. 데이터
import numpy as np

# 각 300개의 데이터를 가지고 있는 x 2개와, 100개의 데이터를 가지고 있는 y
x1 = np.array([range(100), range(311, 411), range(100)])
x2 = np.array([range(101, 201), range(311, 411), range(101, 201)])

y = np.array([range(501, 600)])  # , range(711,811), range(100)])

# 현재 shape은 (3,100)이므로 (100,3)으로 reshape 해주기
x1 = np.transpose(x1)
x2 = np.transpose(y)
y = np.transpose(x2)

# train, test, validation 을 분리하기
from sklearn.model_selection import train_test_split

x1_train, x1_test, y_train, y_test = train_test_split(
    x1, y, random_state=66, test_size=0.4, shuffle=False
)
x1_val, x1_test, y_val, y_test = train_test_split(
    x1_test, y_test, random_state=66, test_size=0.5, shuffle=False
)
x2_train, x2_test = train_test_split(x2, random_state=66, test_size=0.4, shuffle=False)
x2_val, x2_test = train_test_split(
    x2_test, random_state=66, test_size=0.5, shuffle=False
)

# 2. 모델 구성하기
# Input 레이어를 이용해 레이어 만들기
from keras.models import Sequential, Model
from keras.layers import Dense, Input

# model = Sequential()

# 함수형 모델로 input1을 만들기
input1 = Input(shape=(3,))
dense1 = Dense(100, activation="relu")(input1)
dense1_2 = Dense(30)(dense1)
dense1_3 = Dense(7)(dense1_2)

# 함수형 모델로 input2 만들기
input2 = Input(shape=(3,))
dense2 = Dense(50, activation="relu")(input2)
dense2_2 = Dense(7)(dense2)

# merge하기
from keras.layers.merge import concatenate

merge1 = concatenate([dense1_3, dense2_2])

# Concatenate() 이용하기
from keras.layers.merge import Concatenate

merge1 = Concatenate()([dense1_3, dense2_2])

# 레이어 엮어주기
model1 = Dense(10)(merge1)
model2 = Dense(5)(model1)
ouput = Dense(1)(model2)  # 아웃풋 레이어

# Model() 사용해 모델 정의하기
model = Model(inputs=[input1, input2], outputs=output)
model.summary()  # 모델 구성 확인하기

# 3. 훈련하기
model.compile(loss="mse", optimizer="adam", metrics=["mse"])

model.fit(
    [x1_train, x2_train],
    y_train,
    epochs=100,
    batch_size=1,
    validation_data=([x1_val, x2_val], y_val),
)

# 평가 예측 후, 지표로 모델 판단하기
# 4. 평가 예측 하기
mse = model.evaluate([x1_test, x2_test], y_test, batch_size=1)
print("mse : ", mse)

# x의 입력 데이터가 2개이므로 list 형태를 취해야 함
y_predict = model.predict([x1_test, x2_test])

for i in range(len(y_predict)):
    print(y_test[i], y_predict[i])
