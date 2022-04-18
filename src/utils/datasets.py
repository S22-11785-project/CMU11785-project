import torch
import pandas as pd
import numpy as np
import os

# # NOTE: work in progress
# # NOTE: check other dataset options
# class UbiquantDataset(torch.utils.data.Dataset):
#     """ Torch Dataset for RNN models
#     """
#     def __init__(
#         self,
#         df_raw,            # The raw dataframe
#         seq_len=30,        # lookback steps
#         horizon=1,         # output the next n steps
#         one_hot = True,    # whether to one-hot encode investment_id 
#         investment_id=None # individual investment_id
#     ):
#         if investment_id is not None:
#             assert investment_id in df_raw['investment_id']
#             df_raw = df_raw.loc[df_raw['investment_id'] == investment_id]
#             df_raw = df_raw.drop(['row_id', 'investment_id'], axis=1)
#         else:
#             # NOTE: one-hot encode the investment_id
#             if one_hot:
#                 df_ivst = pd.get_dummies(df_raw['investment_id'], sparse=True)
#                 df_raw = pd.concat([df_raw, df_ivst], axis=1)
#             df_raw = df_raw.drop(['row_id'], axis=1)

#         self.seq_len, self.horizon = seq_len, horizon
#         self.X, self.Y = df_raw.to_numpy(), df_raw['target'].to_numpy()
#         assert self.X.shape[0] == len(self.Y) # check length of all data
#         # NOTE: consider add padding
#         self.length = self.X.shape[0] - seq_len - horizon + 1
#         assert self.length > 0

#     def __len__(self):
#         return self.length

#     def __getitem__(self, i):
#         xx = np.array(self.X[i:i+self.seq_len], dtype=float)
#         yy = np.array([self.Y[i+self.seq_len+self.horizon-1]], dtype=float)
#         return xx, yy

class UbiquantDatasetOriginal(torch.utils.data.Dataset):
    """ Simple dataset with optional one-hot encoding for investment_id
    """
    def __init__(self, df_raw, invst_ids, one_hot_invest=True, test=False):
        self.test = test # Whether this dataset will be used for testing
        # Handle feature columns
        if one_hot_invest:
            assert invst_ids is not None
            df_raw = df_raw.reset_index(drop=True)
            df_invst = pd.get_dummies(df_raw['investment_id'], dtype=float).reset_index(drop=True)
            df_complete = pd.DataFrame(np.zeros([len(df_invst.index), len(invst_ids)]), columns=invst_ids)
            df_invst = df_invst.combine_first(df_complete)
            df_invst.columns = [f"invst_id_{col}" for col in df_invst.columns]
            df_raw = pd.concat([df_raw, df_invst], axis=1)
            f_cols = df_invst.columns.tolist() + [f'f_{i}' for i in range(300)]
        else:
            f_cols = ['investment_id'] + [f'f_{i}' for i in range(300)]


        df_raw = df_raw.astype('float32')
        info_cols = ['row_id', 'time_id']
        self.X = df_raw[f_cols].to_numpy()
        self.info = df_raw[info_cols].to_numpy()

        if not test:
            t_cols = ['target']
            self.Y = df_raw[t_cols].to_numpy()

        self.length, self.n_features = self.X.shape[0], self.X.shape[1]

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        if self.test:
            return self.X[ind]
        else:
            return self.X[ind], self.Y[ind]

    def get_info(self, ind):
        return self.info[ind]




class UbiquantDatasetByTime(torch.utils.data.Dataset):
    """ Dataset for RNN with all investment_id.
        Need to download the pre-processed by_steps.zip from: 
        https://drive.google.com/file/d/1hpC4Lxf-MnRUDExFnDI7D8pihQFvUQR5/view?usp=sharing
    """
    def __init__(self,
                 dir_steps,
                 dir_target_parquet,
                 lookback=7,
                 horizon=1):

        self.X_dir = dir_steps
        self.Y_dir = dir_target_parquet
        self.X_files = os.listdir(self.X_dir) # unsorted
        self.Y_df = pd.read_parquet(dir_target_parquet)
        self.time_ids = sorted([int(s.split('.')[0]) for s in self.X_files])
        self.invst_ids = [int(col.split('_')[0]) for col in self.Y_df.columns]
        self.length = len(self.time_ids) - lookback - horizon + 1
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        # TODO: check if we can improve performance by reading parquet without pandas
        ls_x = [pd.read_parquet(os.path.join(self.X_dir, f"{self.time_ids[i]}.parquet"),
                                engine='fastparquet') for i in range(ind, ind+self.lookback)]
        xx = np.array(ls_x).squeeze()
        yy = self.Y_df.iloc[ind+self.lookback+self.horizon-1].to_numpy()
        return xx, yy
    
class UbiquantDatasetByInvestmentID(torch.utils.data.Dataset):
    """ Dataset for RNN with all investment_id.
        pre-processed data
    """
    def __init__(self, data_path, partition= "train"):

        self.X_dir = data_path + "/feats/" 
        self.Y_dir = data_path + "/target/" 

        self.X_files = os.listdir(self.X_dir)   # TODO: list files in the feats directory
        self.Y_files = os.listdir(self.Y_dir)   # TODO: list files in the target directory
       
        assert(len(self.X_files) == len(self.Y_files))

    def __len__(self):
        return len(self.X_files)

    def __getitem__(self, ind):

        X_path = self.X_dir + self.X_files[ind]
        Y_path = self.Y_dir + self.Y_files[ind]
            
        X = torch.from_numpy(np.load(X_path)) #(T, num_feat)
        Y = torch.from_numpy(np.load(Y_path)) #(T,)

        return X, Y
