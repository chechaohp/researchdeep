McNugget_amount = (50,51,52,53,54,55,56,57,58,59,60,61)

for amount in McNugget_amount:
    for pack6_num in range(0,amount//6+1):
        for pack9_num in range(0,amount//9+1):
            for pack20_num in range(0,amount//20+1):
                if pack6_num*6+pack9_num*9+pack20_num*20 == amount:
                    print(amount,'pack6: '+str(pack6_num),'pack9: '+str(pack9_num),'pack20: '+str(pack20_num))