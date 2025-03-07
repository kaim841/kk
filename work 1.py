def count():
    n= int(input("请输入一个整数"))
    sum1=0
    sum2=1
    for i in range(1,n+1):
        sum1+=i
        sum2=sum2*i
    print("和为%d,积为%d"%(sum1,sum2))
count()

