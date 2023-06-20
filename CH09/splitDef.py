# split()함수 다:1 만들기

# 1. 데이터
import numpy as np

dataset = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


def split_xy1(dataset, time_steps):  # (데이터셋, 몇 개의 컬럼으로 자를 지 개수)
    x, y = list(), list()  # 리턴해줄 x, y를 리스트로 선언
    for i in range(len(dataset)):  # 데이터셋의 개수만큼 for문 돌리기
        end_number = i + time_steps  # end_number가 몇인지를 정의하기
        if end_number > len(dataset) - 1:  # 전체 길이에서 1을 뺀 값보다 크면 for문을 정지
            break
        tmp_x, tmp_y = (
            dataset[i:end_number],
            dataset[end_number],
        )  # end_number가 10이 넘지 않을 때까지 반복해서 리스트에 append로 추가
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)


x, y = split_xy1(dataset, 4)  # 데이터를 4개씩 자르기
print(x, "\n", y)

"""
[[1 2 3 4]
 [2 3 4 5]
 [3 4 5 6]
 [4 5 6 7]
 [5 6 7 8]
 [6 7 8 9]]
 [ 5  6  7  8  9 10]
"""
