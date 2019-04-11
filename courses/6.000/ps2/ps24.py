###
### template of code for Problem 4 of Problem Set 2, Fall 2008
###

bestSoFar = 0     # variable that keeps track of largest number
                  # of McNuggets that cannot be bought in exact quantity
packages = (50,20,24)   # variable that contains package sizes

for n in range(1, 150):   # only search for solutions up to size 150
    ## complete code here to find largest size that cannot be bought
    ## when done, your answer should be bound to bestSoFar
    for a in range(0,n//packages[0]+1):
        for b in range(0,n//packages[1]+1):
            for c in range(0,n//packages[2]+1):
                if a*packages[0]+b*packages[1]+c*packages[2] == n:
                    # print(a,b,c)
                    bestSoFar = n
print(bestSoFar)