import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional

class CovariateDatasetBenchmark(Dataset):
    def __init__(
        self,
        size: dict=None,
        scale: bool = True,
        nonautoregressive: bool = False,
        data_path: str = './dataset/ETT-small/ETTh1.csv',
        flag: str = 'train',
        target_columns: Optional[list] = None,
        data_split: Optional[dict] = {'train': 0.7, 'valid': 0.1, 'test': 0.2},
        clean: Optional[bool] = False,
        shift: Optional[int] = 0,
    ):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.pred_len = size[3]
        self.token_num = self.seq_len // self.input_token_len

        self.flag = flag
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.dataset_file_path = data_path
        self.data_type = os.path.basename(self.dataset_file_path).lower()

        self.scale = scale
        self.shift = shift
        self.data_split = data_split

        self.target_columns = target_columns

        self.mean_target = None
        self.std_target = None
        self.mean_cov = None
        self.std_cov = None

        self.__read_data__()

    def __read_data__(self):
        if self.dataset_file_path.endswith(".csv"):
            df_raw = pd.read_csv(self.dataset_file_path)
        elif self.dataset_file_path.endswith(".parquet"):
            df_raw = pd.read_parquet(self.dataset_file_path)
        else:
            raise ValueError("Unknown data format")

        if isinstance(df_raw[df_raw.columns[0]].iloc[0], str):
            # 第一列是 date
            self.time_all = pd.to_datetime(df_raw[df_raw.columns[0]])
            df_raw = df_raw[df_raw.columns[1:]]
        else:
            self.time_all = None


        if self.target_columns is None:
            raise ValueError("target_columns must be provided")

        all_cols = list(self.target_columns)
        covar_cols = all_cols[:-1]      # 协变量
        tgt_col = all_cols[-1]          # 最后一列目标

        # 提取数值矩阵
        covariate_all = df_raw[covar_cols].values.astype(float)  # [T, C_cov]
        target_all = df_raw[[tgt_col]].values.astype(float)      # [T, 1]

        if "etth" in self.data_type:
            border1s = [0,
                        12 * 30 * 24 - self.seq_len,
                        12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24,
                        12 * 30 * 24 + 4 * 30 * 24,
                        12 * 30 * 24 + 8 * 30 * 24]
        elif "ettm" in self.data_type:
            border1s = [0,
                        12 * 30 * 24 * 4 - self.seq_len,
                        12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4,
                        12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                        12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            data_len = len(df_raw)
            num_train = int(data_len * self.data_split['train'])
            num_test = int(data_len * self.data_split['test'])
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.seq_len, data_len - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, data_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.time_all is not None:
            self.time_split = self.time_all[border1:border2].reset_index(drop=True)
        else:
            self.time_split = None

        if self.scale:
            # target normalize
            train_tgt = target_all[border1s[0]:border2s[0]]
            self.mean_target = np.mean(train_tgt, axis=0)
            self.std_target = np.std(train_tgt, axis=0) + 1e-5
            target_all = (target_all - self.mean_target) / self.std_target

            # cov normalize
            train_cov = covariate_all[border1s[0]:border2s[0]]
            self.mean_cov = np.mean(train_cov, axis=0)
            self.std_cov = np.std(train_cov, axis=0) + 1e-5
            covariate_all = (covariate_all - self.mean_cov) / self.std_cov

        self.data_tgt = target_all[border1:border2]         # [T_split, 1]
        self.data_cov = covariate_all[border1:border2]      # [T_split, C_cov]

        self.n_time = len(self.data_tgt)

    def __getitem__(self, index):
        """返回:
            seq_x      : 历史目标
            seq_x_cov  : 历史协变量
            seq_y      : 未来目标
            seq_y_cov  : 未来协变量
        """
        s_begin = index
        s_end = s_begin + self.seq_len

        # 和大模型预训练不同，这里没有切分token，直接按照测试集方式划分训练集
        r_begin = s_end
        r_end = r_begin + self.pred_len

        # 历史
        seq_x = self.data_tgt[s_begin:s_end]           # [L_hist, 1]
        seq_x_cov = self.data_cov[s_begin:s_end]       # [L_hist, C_cov]

        # 未来（label）
        seq_y = self.data_tgt[r_begin:r_end]           # [L_pred, 1]
        seq_y_cov = self.data_cov[r_begin:r_end]       # [L_pred, C_cov]

        return seq_x, seq_x_cov, seq_y, seq_y_cov


    def __len__(self):
        return self.n_time - self.seq_len - self.pred_len + 1

    def inverse_transform_target(self, data):
        mean = torch.tensor(self.mean_target, dtype=data.dtype, device=data.device)
        std = torch.tensor(self.std_target, dtype=data.dtype, device=data.device)
        return data * std + mean
