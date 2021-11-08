tried = 0
a = []
c = 0
while True:
    num = int(input("정수 입력: "))
    if num != -1 and 2< num <100000:
        a.append(num)
        tried += 1
    elif num == -1:
        break
    else:
        print("너무 크거나 작습니다!")

for i in range(tried):
    a[0] = d
    for j in range(d):
        if d % j == 0:
            b = b + str(j)
            c = c + j
    if c == d:
        print(str(d)+" = " + b)
    else:
        print(str(d)+ "is Not perfect")
