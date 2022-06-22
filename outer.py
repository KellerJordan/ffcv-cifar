import os
import random
#import multiprocessing as mp
import torch
import torch.multiprocessing as mp

import train

def get_args_list():
    args_list = []
    for epochs in [1, 2, 4, 8, 16, 32, 64, 128]:
        for _ in range(10):
            args_d = {
                'lr': 0.5,
                'batch_size': 512,
                'epochs': epochs,
                'save_outputs': 1,
                'num_runs': 50,
            }
            args_list.append(args_d)
    random.shuffle(args_list)
    return args_list

class Args:
    def __init__(self, args_d):
        self.__dict__ = args_d

def run_all(gpu, ngpus, args_list):
    for i, args_d in enumerate(args_list):
        if i % ngpus == gpu:
            print(args_d)
            args_d['gpu'] = gpu
            train.main(Args(args_d))

if __name__ == '__main__':
    args_list = get_args_list()
    ngpus = torch.cuda.device_count()
    mp.spawn(run_all, nprocs=ngpus, args=(ngpus, args_list))

