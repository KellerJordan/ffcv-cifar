import os
import uuid
import pickle
import argparse

from tqdm import tqdm
import numpy as np

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
import torchvision

from model import create_model
from data import create_loader

def evaluate(model, test_loader, save_outputs=False):
    model.eval()
    stats = {
        'correct': 0,
    }
    outputs_l = []
    with torch.no_grad(), autocast():
        for batch in test_loader:
            inputs, labels = batch
            outputs = model(inputs)
            if save_outputs:
                outputs_l.append(outputs)
            pred = outputs.argmax(dim=1)
            stats['correct'] += (labels == pred).sum().item()

    if save_outputs:
        outputs = torch.cat(outputs_l).cpu().numpy()
        stats['outputs'] = outputs
    return stats

def train(args, verbose=True):

    train_loader = create_loader(args.batch_size, True, args.gpu)
    test_loader = create_loader(1024, False, args.gpu)

    n_iters = len(train_loader)
    lr_schedule = np.interp(np.arange(1+n_iters), [0, n_iters], [1, 0])

    model = create_model().cuda(args.gpu)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss()

    log = {
        'args': args.__dict__,
        'losses': [],
        'corrects': [],
    }
    it = range(args.epochs)
    if verbose:
        it = tqdm(it)
    for epoch in it:
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
        
            scaler.scale(loss).backward()
            log['losses'].append(loss.item())
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        if (epoch+1) % 12 == 0:
            c = evaluate(model, test_loader)['correct']
            if verbose:
                it.set_postfix({'correct': c})
            log['corrects'].append(c)

    if args.save_outputs:
        stats = evaluate(model, test_loader, save_outputs=True)
        log['correct'] = stats['correct']
        if verbose:
            print('correct=%d' % stats['correct'])
        log['outputs'] = stats['outputs']

    os.makedirs('./logs', exist_ok=True)
    log_path = os.path.join('./logs', str(uuid.uuid4())+'.pkl')
    with open(log_path, 'wb') as f:
        pickle.dump(log, f)

def main(args):
    many_runs = (args.num_runs >= 10)
    it = range(args.num_runs)
    if many_runs:
        it = tqdm(it)
    try:
        for _ in it:
            train(args, verbose=not many_runs)
    except KeyboardInterrupt:
        pass

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=48)
parser.add_argument('--save_outputs', type=int, default=1)
parser.add_argument('--num_runs', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

