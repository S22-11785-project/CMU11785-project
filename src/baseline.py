import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import wandb
from datetime import datetime
import time
from sklearn.model_selection import train_test_split

# local
import utils.utils as utils
import utils.datasets as datasets
import models.models as models


DIR_REPO = '/media/user/12TB1/HanLi/GitHub/CMU11785-project'
DIR_DATA = os.path.join(DIR_REPO, 'local_data/archive')
DIR_TRAINED = os.path.join(DIR_REPO, 'local_trained/20220401')
print("==>> DIR_REPO: ", DIR_REPO)
print("==>> DIR_DATA: ", DIR_DATA)
print("==>> DIR_TRAINED: ", DIR_TRAINED)

if not os.path.exists(DIR_TRAINED):
    os.makedirs(DIR_TRAINED)


################################################################################
def main(args):
    print('==>> Start main loop...')
    print('==>> Loading data...')
    df_raw = pd.read_parquet(os.path.join(DIR_DATA, 'train_low_mem.parquet'))
    invst_ids = sorted(list(df_raw['investment_id'].unique()))
    if 'subset' in args: 
        df_raw = df_raw.iloc[:int(args['subset']*len(df_raw.index))]

    df_train, df_eval = train_test_split(df_raw, test_size=0.25, shuffle=True)
    batch_size = args['batch_size']
    train_dataset = datasets.UbiquantDatasetOriginal(df_train, invst_ids, one_hot_invest=False)
    val_dataset = datasets.UbiquantDatasetOriginal(df_eval, invst_ids, one_hot_invest=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print('==>> Create model...')
    model = models.MLP(in_size=train_dataset.n_features).cuda()
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * args['epochs']), eta_min=args['min_lr'])
    scaler = torch.cuda.amp.GradScaler()

    args['train_loader'] = train_loader
    args['val_loader']   = val_loader
    args['criterion']    = criterion
    args['optimizer']    = optimizer
    args['scheduler']    = scheduler
    args['scaler']       = scaler

    if 'wb_project' in args:
        wandb.init(
            entity=args['wb_entity'],
            project=args['wb_project'],
            name=args['wb_name'],
            config=args # NOTE: log our args and configs
        )

    print('==>> Start training...')
    for epoch in range(args['starting_epoch'], args['epochs']+args['starting_epoch']):
        a = datetime.now()
        args['epoch'] = epoch
        train_loss = utils.train_single_epoch(args, model)
        val_loss, pearsonr = utils.evaluate(args, model)
        lr_o = round(float(optimizer.param_groups[0]['lr']), 5)
        b = datetime.now()
        e_time = round((b-a).total_seconds(), 1)
        print(f"[{time.asctime()}] - Epoch {epoch}/{args['epochs']}: Train Loss {train_loss}, Val Loss {val_loss}, Pearson R {pearsonr}, Learning Rate {lr_o}, finished in {e_time} seconds.")
        if 'wb_project' in args: 
            wandb.log({"Train Loss": train_loss, "Val Loss": val_loss, "Pearson R":pearsonr, "Time Used (s)": e_time, "lr": lr_o})

    print('==>> Training completes')
    torch.save(model, os.path.join(DIR_TRAINED, 'model_name.pt'))

    if 'wb_project' in args: wandb.finish()

################################################################################

if __name__ == '__main__':
    args = {
        # 'subset': 0.1, # portion of data to use, comment this line to use full data
        'batch_size': 256,
        'starting_epoch': 1,
        'epochs': 15,
        'lr': 1e-3,
        'min_lr': 1e-4,
        'wb_entity': '11785_project',
        'wb_project': 'baseline',
        'wb_name': 'baseline_MLP',
    }
    main(args)