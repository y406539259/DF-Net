import os

#For Drive dataset.
# Dir = "/home/yinpengshuai/Desktop/FrangiNet/data/CHASEDB1/testing"
# allfile = os.listdir(Dir)
# with open("/home/yinpengshuai/Desktop/FrangiNet/data/CHASEDB1/CHASEDB1testing.txt", "w") as f:
#     for i in allfile:
#         f.write(i)
#         f.write('\n')

#For Stare dataset.
Dir = "/home/yinpengshuai/Desktop/FrangiNet/data/Stare/training"
allfile = os.listdir(Dir)
count = 0
for j in allfile:
    count = count + 1
    with open("/home/yinpengshuai/Desktop/FrangiNet/data/Stare/Staretesting" + str(count) + '.txt', "w") as f:
            f.write(j)
            f.write('\n')

    with open("/home/yinpengshuai/Desktop/FrangiNet/data/Stare/Staretraining" + str(count) + '.txt', "w") as f:
        for i in allfile:
            if i !=j:
                f.write(i)
                f.write('\n')

