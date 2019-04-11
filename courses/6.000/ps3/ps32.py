def subStringMatchExact(target,key):
    position = ()
    for i in range(0,len(target)-len(key)):
        if target[i:i+len(key)] == key:
            position = position + (i,)
    return position

target1 = 'atgacatgcacaagtatgcat'
target2 = 'atgaatgcatggatgtaaatgcag'

key10 = 'a'
key11 = 'atg'
key12 = 'atgc'
key13 = 'atgca'

print(subStringMatchExact(target1,key10))
print(subStringMatchExact(target1,key11))
print(subStringMatchExact(target1,key12))
print(subStringMatchExact(target1,key13))

print(subStringMatchExact(target2,key10))
print(subStringMatchExact(target2,key11))
print(subStringMatchExact(target2,key12))
print(subStringMatchExact(target2,key13))
