import os
import random
import subprocess
import multiprocessing as mp

from train import main as run_train

def get_args_list():
    args_list = []
    for epochs in [16]:
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

def run_all(gpu, arg_list):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    for args_d in arg_list:
        print(args_d)
        run_train(Args(args_d))

if __name__ == '__main__':
    ctx = mp.get_context('spawn')

    arg_lists = [[] for _ in range(8)]
    for i, arg_d in enumerate(get_args_list()):
        arg_lists[i % 8].append(arg_d)

    proc_l = []
    for i in range(8):
        p = ctx.Process(target=run_all, args=(i, arg_lists[i]))
        p.start()
        proc_l.append(p)

    for p in proc_l:
        p.join()

