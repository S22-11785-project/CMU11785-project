import os
import torch
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import wandb
from datetime import datetime
import time

# local
import utils.utils as utils
import models


DIR_REPO = '/media/user/12TB1/HanLi/GitHub/CMU11785-project/'
DIR_DATA = os.path.join(DIR_REPO, '/local_data/archive')
DIR_TRAINED = os.path.join(DIR_REPO, '/local_trained/20220401')

if not os.path.exists(DIR_TRAINED):
    os.makedirs(DIR_TRAINED)


################################################################################
def main(args):
    df_raw = pd.read_parquet(os.path.join(DIR_DATA, 'train_low_mem.parquet'))

    # TODO: building train and evaluate datasets and data loaders
    train_loader = None
    eval_loader = None

    model = models.MLP()
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * args['num_epochs']), eta_min=args['min_lr'])
    scaler = torch.cuda.amp.GradScaler()

    args['criterion'] = criterion
    args['optimizer'] = optimizer
    args['scheduler'] = scheduler
    args['scaler']    = scaler

    if 'wb_project' in args:
        wandb.init(
            entity='tsbyq_wb',
            project=args['wb_project'],
            name=args['wb_name'],
            config=args # NOTE: log our args and configs
        )
    for epoch in range(args['starting_epoch'], args['epochs']+args['starting_epoch']):
        a = datetime.now()
        train_loss = utils.train_single_epoch(args, model)
        val_loss = utils.evaluate(args, model)
        lr_o = round(float(optimizer.param_groups[0]['lr']), 5)
        b = datetime.now()
        e_time = round((b-a).total_seconds(), 1)
        print(f"[{time.asctime()}] - Epoch {epoch}/{args['num_epochs']}: Train Loss {train_loss}, Val Loss {val_loss}, Learning Rate {lr_o}, finished in {e_time} seconds.")
        if 'wb_project' in args: 
            wandb.log({"Train Loss": train_loss, "Val Loss": val_loss, "Time Used (s)": e_time, "lr": lr_o})

    torch.save(model, os.path.join(DIR_TRAINED, 'model_name.pt'))

    if 'wb_project' in args: wandb.finish()

################################################################################

if __name__ == '__main__':
    args = {
        'starting_epoch': 1,
        'epochs': 15,
        'lr': 1e-3,
        'wb_entity': '11785-project',
        'wb_project': 'baseline',
        'wb_name': 'baseline_test',
    }
    main(args)