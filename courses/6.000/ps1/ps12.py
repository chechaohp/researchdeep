import math

print('enter your number:')
n = 1000
if int(n) < 2:
    print('n must be positive integer and larger than 2')
    print('Exit')
else:
    sum_log = math.log(2)
    current_number = 2
    while (current_number <= int(n)):
        print(current_number,end = '\r')
        divide_number_check = 2
        while (divide_number_check < current_number):
            if current_number%divide_number_check==0:
                break
            if divide_number_check == current_number - 1:
                sum_log += math.log(current_number)
            divide_number_check += 1
        current_number += 1
    print(sum_log/int(n))