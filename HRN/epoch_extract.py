from turtle import shape
import numpy as np

list_path = "epochs.txt"
with open(list_path, 'r') as f:
    for l in f.readlines():
        # if l.startswith("Epoch"):
        #     lists_epoch = l.strip('\n').strip(',')
        #     print(lists_epoch[-3:])
            


        # if l.startswith("Iteration"):
        #     lists_iteration = l.strip('\n').split(',')
        #     # print(lists_iteration[1][-8:])
        #     print(lists_iteration[5][-8:])

        if l.startswith("test_order_acc"):
            lists_test = l.strip('\n').split(',')
            print(lists_test[4][-8:])


        # lists = l.strip('\n').split(',')
        # if lists[0]!="\n":
        #     print(list(lists[0]))