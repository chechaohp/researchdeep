from string import *

s = 'atgacatgcaagtatgcat'

def countSubStringMatch(target,key):
    """This function is use for counting substring"""
    count = 0
    for i in range(0,len(target)-len(key)):
        if target[i:i+len(key)] == key:
            count += 1
    return count

def countSubStringMatchRecursive(target,key):
    find_string = target.find(key)
    if find_string == -1:
        return 0
    else:
        return countSubStringMatchRecursive(target[find_string+1:],key) + 1

print(countSubStringMatch(s,'atg'))
print(countSubStringMatchRecursive(s,'atg'))
