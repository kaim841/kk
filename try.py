class dog():

    def __init__(self,name,type,color):
        self.name=name
        self.color=color
        self.type=type

    def roll(self,food):
        print(self.name,"高兴的转圈","吃了",food)

    def skip(self):
        self.wheel=50
        print(self.name,"跳起来了")

dog1=dog(name="佳佳",color="蓝色",type="嘎嘎")
dog1.roll(food="豆")




print(dog1.skip())