def calc_sum(x,y):
    s = x + y
    return(s)

def sum_diff2(x,y):
    s = x + y
    d = x - y
    return(s,d)

def calculate(x,y,option):
    if (option == "sum"):
        res = x + y
    elif (option == "diff"):
        res = x - y
    elif (option == "mult"):
        res = x*y
    elif (option == "div"):
        res = x/y
    else:
        print("unknown option. returning zero")
        res = 0
    return(res)

polynom_l = lambda x: x**2 + 5*x + 4

polynom_l2 = lambda x=0,y=0: x**3 + 3*x*y + y**2 + 15
