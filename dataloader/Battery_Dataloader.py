
from src.Warehouse import *
from torch.utils.data import DataLoader
from src.timefeatures import time_features

class Dataset_loader(Dataset):
    def __init__(self, data_path, flag, seq_len, label_len, pred_len,freq, timeenc=1, seasonal_patterns=None):

        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.timeenc = timeenc
        self.freq = freq #时间编码
        self.scaler = StandardScaler() #归一化
        self.df_raw = pd.read_csv(data_path)#载入数据

        self.datazize = self.df_raw.shape[0]
        border1s = [0,
                    int(0.6*self.datazize),
                    int(0.8*self.datazize)]

        border2s = [int(0.6*self.datazize),
                    int(0.8*self.datazize),
                    int(self.datazize)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = self.df_raw.columns[-1:]
        df_data = self.df_raw[cols_data]

        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)
        df_stamp = self.df_raw[['Cycle']][border1:border2] #时间列

        df_stamp['Cycle'] = pd.to_datetime(df_stamp.Cycle)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['Cycle'], 1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['Cycle'].values), self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]#可以修改

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):

        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def data_provider(embed, data_path, flag, seq_len, label_len, pred_len,freq, timeenc=0):

    timeenc = 0 if embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        batch_size = 1 # bsz=1 for evaluation
    else:
        shuffle_flag = False
        batch_size = 4  # bsz for train and valid

    drop_last = True
    data_set = Dataset_loader(
        data_path = data_path,
        flag = flag, seq_len = seq_len, label_len = label_len, pred_len = pred_len, freq = freq,
        timeenc=1, seasonal_patterns=None
        )

    data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=4,
            drop_last=drop_last)
    return data_set, data_loader

def battery_loader(embed, data_path, seq_len, label_len, pred_len, freq, flag):

    data_set, data_loader = data_provider(embed,data_path, flag, seq_len, label_len, pred_len,freq)

    return data_set, data_loader
