

def nestEggFixed(salary,save,growthRate,years):
    retirement_account = []
    for i in range(0,years):
        if i == 0:
            retirement_account += [salary*save*0.01]
        else:
            retirement_account += [retirement_account[i-1]*(1+0.01*growthRate)+salary*save*0.01]
    return retirement_account

print(nestEggFixed(10000,0.05,0.05,10))