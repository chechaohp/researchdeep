highest = 0
solution_count = 0
while (solution_count < 6):
    highest += 1
    all_has_solution_check = False
    nugget_check = [highest +1,highest + 2, highest+3,highest+4,highest+5,highest+6]
    solution_count = 0
    for amount in nugget_check:
        for pack6_num in range(0,amount//6+1):
            for pack9_num in range(0,amount//9+1):
                for pack20_num in range(0,amount//20+1):
                    if pack6_num*6+pack9_num*9+pack20_num*20 == amount:
                        # print(1)
                        solution_count +=1
                        break
                if pack6_num*6+pack9_num*9+pack20_num*20 == amount:
                    # print(2)
                    # solution_count +=1
                    break
            if pack6_num*6+pack9_num*9+pack20_num*20 == amount:
                print(amount,'pack6: '+str(pack6_num),'pack9: '+str(pack9_num),'pack20: '+str(pack20_num))
                break
    print('solution count:',solution_count)
    print('highest number:',highest)
                        