class Student:
    def __init__(self,kor,mat,eng):
        self.kor = kor
        self.mat = mat
        self.eng = eng

    def get_ave(self):
        return((self.kor+self.mat+self.eng)/3)

s = Student(95,90,100)
print(s.get_ave())