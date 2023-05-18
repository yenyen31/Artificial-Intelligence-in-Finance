# 5장. 앙상블
# 2개 이상의 모델을 합치는 방법

# 1. 데이터
import numpy as np

x1 = np.array([range(100), range(311, 411), range(100)])
x2 = np.array([range(101, 201), range(311, 411), range(100)])
y = np.array([range(501, 601)])  # , range(711, 811), range(100)])

# 각 데이터는 3개의 컬럼을 가지고 있음
# 현재 shape은 (3,100)이므로 (100,3)으로 reshape해야 함
x1 = np.transpose(x1)
x2 = np.transpose(x2)
y = np.transpose(y)

# train, test, validation 분리하기
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

# 2. 모델 구성
# 2개의 모델을 만든 뒤, 2개의 모델을 다시 병합하기 -> 앙상블 모델
from keras.models import Sequential, Model
from keras.layers import Dense, Input  # Input레이어를 이용해 레이어 만들기

# 함수형 모델로 input1 생성하기
input1 = Input(shape=(3,))
dense1 = Dense(100, activation="relu")(input1)
dense1_2 = Dense(30)(dense1)
dense1_3 = Dense(7)(dense1_2)

# input2 생성하기
input2 = Input(shape=(3,))
dense2 = Dense(50, activation="relu")(input2)
dense2_2 = Dense(7)(dense1_2)

# merge 부분 (가장 중요!)
from keras.layers.merge import concatenate

# concatenate() 매개변수로 리스트 방식으로 연결해주기
merge1 = concatenate([dense1_3, dense2_2])

# merge 이후에는 다음 레이어를 차례대로 엮어주면 됨
model1 = Dense(10)(merge1)
model2 = Dense(5)(model1)
output = Dense(1)(model2)

# Model() 사용해 만든 모델들 정의하기
model = Model(inputs=[input1, input2], outputs=output)
model.summary()  # 모델 구성 확인하기\


# 3. 훈련
# x의 입력 데이터가 2개이므로 리스트 형태를 취해야 함
# [x1_test, x2_test]와 Validation 모두 list 형태로 입력해야 함 주의!

model.compile(loss="mse", optimizer="adam", metrics=["mse"])
model.fit(
    [x1_train, x2_train],
    y_train,
    epochs=100,
    batch_size=1,
    validation_data=([x1_val, x2_val], y_val),
)

# 4. 평가 예측
# 평가 시에도 리스트 형태를 취해야 함
mse = model.evaluate([x1_test, x2_test], y_test, batch_size=1)
print("mse : ", mse)

y_predict = model.predict([x1_test, x2_test])

for i in range(len(y_predict)):
    print(y_test[i], y_predict[i])
