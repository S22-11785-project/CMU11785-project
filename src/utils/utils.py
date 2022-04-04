import torch
from tqdm import tqdm
import os


def train_single_epoch(args, model):
    # Train a single epoch
    train_loader = args['train_loader']
    criterion = args['criterion']
    optimizer = args['optimizer']
    lr_scheduler = args['scheduler']
    scaler = args['scaler']
    epoch = args['epoch']
    
    model.train()
    total_loss = 0
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True,
                     leave=False, position=0, desc=f"Train epoch {epoch}")
    for b_idx, data in enumerate(train_loader):
        x, y = data
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            y_pred = model.forward(x)
            loss = criterion(y_pred, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss)
        batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss/(b_idx + 1))),
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr']))
        )
        batch_bar.update()
        lr_scheduler.step()
    total_loss /= (len(train_loader)+1)
    batch_bar.close()
    return total_loss

def evaluate(args, model):
    epoch = args['epoch']
    criterion = args['criterion']
    val_loader = args['val_loader']
    model.eval()

    val_loss = 0
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True,
                     leave=False, position=0, desc=f"Evaluate epoch {epoch}")
    for b_idx, data in enumerate(val_loader, 0):
        x, y = data
        x, y = x.cuda(), y.cuda()
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                y_pred = model.forward(x)
            loss = criterion(y_pred, y)
            val_loss += float(loss)
        batch_bar.set_postfix(
            loss="{:.04f}".format(float(val_loss/(b_idx + 1))),
        )
    val_loss /= (len(val_loader)+1)
    # Save model as needed
    # if scheduler is not None: scheduler.step(val_loss)
    return val_loss

        