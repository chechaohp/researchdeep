i = 1
print(2)
prime_check = 3
while (i<1000):
     divide_test_number = 2
     while (divide_test_number < prime_check):
          if prime_check%divide_test_number == 0:
               break
          if divide_test_number == (prime_check-1):
               print(i,prime_check)
               i = i+1
          divide_test_number += 1
     prime_check += 1
         