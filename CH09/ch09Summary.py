# split_xy1 (다:1)
# 원데이터
dataset = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# 함수 적용 후
x, y = split_xy1(dataset, 4)

# split_xy2 (다:다)
# 원데이터
dataset = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# 함수 적용 후
x, y = split_xy2(dataset, 4, 2)

# split_xy3 (다입력, 다:1)
# 원데이터
dataset = np.array(
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    ]
)
dataset = np.transpose(dataset)
# 함수 적용 후
x, y = split_xy3(dataset, 3, 1)

# split_xy3 (다입력, 다:다)
# 원데이터
dataset = np.array(
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    ]
)
dataset = np.transpose(dataset)
# 함수 적용 후
x, y = split_xy3(dataset, 3, 2)


# split_xy5 (다입력, 다:다) - 두 번째 버전
# 원데이터
dataset = np.array(
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    ]
)
dataset = np.transpose(dataset)
# 함수 적용 후
x, y = split_xy5(dataset, 3, 1)

# split_xy5 (다입력, 다:다) - 세 번째 버전
# 원데이터
dataset = np.array(
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    ]
)
dataset = np.transpose(dataset)
# 함수 적용 후
x, y = split_xy5(dataset, 3, 2)
