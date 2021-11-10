n = int(input("입력 횟수: "))
b = []
d = 0
for i in range(n):
    a = int(input(str(i+1)+"번째 입력: "))
    b.append(a)

for i in range(n):
    c = str(i+1)+": "
    for j in b:
        if b[i] < j:
            c = c + "<" + " "
        elif b[i] > j:
            c = c + ">" + " "
        elif b[i] == j:
            c = c + "=" + " "
    print(c)