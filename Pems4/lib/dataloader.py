import torch
import numpy as np
import torch.utils.data
from Second_paper_Distribution.Pems4.lib.add_window import Add_Window_Horizon
from Second_paper_Distribution.Pems4.lib.load_dataset import load_st_dataset
from Second_paper_Distribution.Pems4.lib.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler


def normalize_dataset(data, normalizer, column_wise=False, val=0.2, test=0.2):
    # 2022-11-23----------这里修改了归一化，应该是没问题的啊--------------
    data_len = data.shape[0]
    index = int(data_len * (test + val))
    if normalizer == 'max01':
        if column_wise:
            minimum = data[: -index].min(axis=0, keepdims=True)
            maximum = data[: -index].max(axis=0, keepdims=True)
        else:
            minimum = data[: -index].min()
            maximum = data[: -index].max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data[: -index].min(axis=0, keepdims=True)
            maximum = data[: -index].max(axis=0, keepdims=True)
        else:
            minimum = data[: -index].min()
            maximum = data[: -index].max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data[: -index].mean(axis=0, keepdims=True)
            std = data[: -index].std(axis=0, keepdims=True)
        else:
            mean = data[: -index].mean()
            std = data[: -index].std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        # column min max, to be depressed
        # note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data[: -index].min(axis=0), data[: -index].max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler


def split_data_by_days(data, val_days, test_days, interval=60):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24 * 60) / interval)
    test_data = data[-T * test_days:]
    val_data = data[-T * (test_days + val_days): -T * test_days]
    train_data = data[:-T * (test_days + val_days)]
    return train_data, val_data, test_data


def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len * test_ratio):]
    val_data = data[-int(data_len * (test_ratio + val_ratio)):-int(data_len * test_ratio)]
    train_data = data[:-int(data_len * (test_ratio + val_ratio))]
    return train_data, val_data, test_data


def split_support_query(data_train, data_y, split_ratio_support, sequences=True):
    # 这个比例完全就是支持集的分割比例啊
    data_len = data_train.shape[0]
    split_num = int(data_len * split_ratio_support)
    if sequences:
        # 顺序存取
        support_train = data_train[: split_num]
        support_y = data_y[: split_num]

        query_train = data_train[split_num:]
        query_y = data_y[split_num:]
    else:
        # 随机
        index_all = np.arange(data_len)
        index_support = np.random.choice(data_len, split_num, replace=False)
        index_query = np.delete(index_all, index_support)

        support_train = data_train[index_support]
        support_y = data_y[index_support]

        query_train = data_train[index_query]
        query_y = data_y[index_query]

    return support_train, support_y, query_train, query_y


def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def get_dataloader_stamp(args, normalizer='std', tod=False, dow=False, weather=False, single=True,
                         add_time_in_day=True, add_day_in_week=True):
    # load raw st dataset
    print(args)
    data = load_st_dataset(args.dataset)  # B, N, D --pems3的shape是(26208, 358, 1)

    L, N, F = data.shape
    # normalize st data
    raw_data, scaler = normalize_dataset(data, normalizer, args.column_wise, args.val_ratio, args.test_ratio)

    stamp_list = [raw_data]
    if add_time_in_day:
        # numerical time_in_day
        time_ind = [i % args.steps_per_day / args.steps_per_day for i in range(data.shape[0])]
        time_ind = np.array(time_ind)

        time_in_day = np.tile(time_ind, [1, N, 1]).transpose((2, 1, 0))
        stamp_list.append(time_in_day)
    if add_day_in_week:
        # numerical day_in_week
        day_in_week = [(i // args.steps_per_day) % 7 for i in range(data.shape[0])]
        day_in_week = np.array(day_in_week)
        day_in_week = np.tile(day_in_week, [1, N, 1]).transpose((2, 1, 0))
        stamp_list.append(day_in_week)

    data = np.concatenate(stamp_list, axis=-1)

    # spilit dataset by days or by ratio
    if args.test_ratio > 1:
        data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
    else:
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
    # add time window
    x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single)
    x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single)
    x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single)
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)
    ##############get dataloader######################
    train_dataloader = data_loader(x_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)

    return train_dataloader, val_dataloader, test_dataloader, scaler, y_test, data_train
