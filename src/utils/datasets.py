import torch
import pandas as pd
import numpy as np

# NOTE: work in progress
# NOTE: check other dataset options
class UbiquantDataset(torch.utils.data.Dataset):
    """ Torch Dataset for RNN models
    """
    def __init__(
        self,
        df_raw,            # The raw dataframe
        seq_len=30,        # lookback steps
        horizon=1,         # output the next n steps
        one_hot = True,    # whether to one-hot encode investment_id 
        investment_id=None # individual investment_id
    ):
        if investment_id is not None:
            assert investment_id in df_raw['investment_id']
            df_raw = df_raw.loc[df_raw['investment_id'] == investment_id]
            df_raw = df_raw.drop(['row_id', 'investment_id'], axis=1)
        else:
            # NOTE: one-hot encode the investment_id
            if one_hot:
                df_ivst = pd.get_dummies(df_raw['investment_id'], sparse=True)
                df_raw = pd.concat([df_raw, df_ivst], axis=1)
            df_raw = df_raw.drop(['row_id'], axis=1)

        self.seq_len, self.horizon = seq_len, horizon
        self.X, self.Y = df_raw.to_numpy(), df_raw['target'].to_numpy()
        assert self.X.shape[0] == len(self.Y) # check length of all data
        # NOTE: consider add padding
        self.length = self.X.shape[0] - seq_len - horizon + 1
        assert self.length > 0

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        xx = np.array(self.X[i:i+self.seq_len], dtype=float)
        yy = np.array([self.Y[i+self.seq_len+self.horizon-1]], dtype=float)
        return xx, yy



# ------------------------------- Others' work ------------------------------- #
# import numpy as np
# import pandas as pd
# import pytorch_lightning as pl
# import torch
# from torch.utils.data import DataLoader, random_split

# # from constants import FEATURES
# FEATURES = [f'f_{i}' for i in range(300)]

# def collate_fn(datas):
#     prems = [torch.randperm(data[0].size(0)) for data in datas]
#     length = min(data[0].size(0) for data in datas)
#     return [
#         torch.stack([d[i][perm][:length] for d, perm in zip(datas, prems)])
#         for i in range(3)
#     ]


# class MyDataset(torch.utils.data.Dataset):
#     def __init__(self, *tensor_lists) -> None:
#         assert all(len(tensor_lists[0]) == len(
#             t) for t in tensor_lists), "Size mismatch between tensor_lists"
#         self.tensor_lists = tensor_lists

#     def __getitem__(self, index):
#         return tuple(t[index] for t in self.tensor_lists)

#     def __len__(self):
#         return len(self.tensor_lists[0])

# def df_to_input_id(df):
#     return torch.tensor(df['investment_id'].to_numpy(dtype=np.int16),
#                         dtype=torch.int)


# def df_to_input_feat(df):
#     return torch.tensor(df[FEATURES].to_numpy(),
#                         dtype=torch.float32)


# def df_to_target(df):
#     return torch.tensor(df['target'].to_numpy(),
#                         dtype=torch.float32)


# def load_data(path):
#     df = pd.read_parquet(path)
#     groups = df.groupby('time_id')
#     return [
#         groups.get_group(v)
#         for v in df.time_id.unique()
#     ]

# def split(df_groupby_time, split_ratios):
#     ids = [df_to_input_id(df) for df in df_groupby_time]
#     feats = [df_to_input_feat(df) for df in df_groupby_time]
#     targets = [df_to_target(df) for df in df_groupby_time]

#     dataset = MyDataset(ids, feats, targets)

#     lengths = []
#     for ratio in split_ratios[:-1]:
#         lengths.append(int(len(dataset)*ratio))
#     lengths.append(len(dataset) - sum(lengths))

#     return random_split(dataset, lengths)


# class UMPDataModule(pl.LightningDataModule):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args

#         datasets = split(load_data(args.input), args.split_ratios)
#         if len(datasets) == 3:
#             self.tr, self.val, self.test = datasets
#         else:
#             self.tr, self.val = datasets
#             self.test = self.val

#     def train_dataloader(self):
#         return DataLoader(self.tr, batch_size=self.args.batch_size,
#                           num_workers=self.args.workers, shuffle=True,
#                           collate_fn=collate_fn, drop_last=True,
#                           pin_memory=True)

#     def _val_dataloader(self, dataset):
#         return DataLoader(dataset, batch_size=1,
#                           num_workers=self.args.workers, pin_memory=True)

#     def val_dataloader(self):
#         return self._val_dataloader(self.val)

#     def test_dataloader(self):
#         return self._val_dataloader(self.test)
# ------------------------------- Others' work ------------------------------- #