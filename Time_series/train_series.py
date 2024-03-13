import torch
import numpy as np
import argparse
import time
import configparser
from TPGCN.Time_series.util import *
import matplotlib.pyplot as plt
from TPGCN.Time_series.engine import trainer
from torch import nn
from TPGCN.Time_series.util_series import generate_metric, DataLoaderS_stamp

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--dataset', type=str, default='data/exchange_rate.txt', help='dataset')

parser.add_argument('--gcn_bool', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--addaptadj', type=int, default=1, help='whether add adaptive adj')

parser.add_argument('--seq_length', type=int, default=168, help='')
parser.add_argument('--nhid', type=int, default=16, help='')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=8, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')  # 0.001
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0., help='weight decay rate')  # 0.0001
parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--seed', type=int, default=99, help='random seed')
parser.add_argument('--save', type=str, default='./model/pems4', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
parser.add_argument('--log_file', type=str, default='./log/pems4_log', help='log file')
parser.add_argument('--headnum', type=int, default=4, help='Number of heads')
parser.add_argument('--rate', type=int, default=1, help='')  # 10
parser.add_argument('--normalize', type=int, default=2,)
parser.add_argument('--dropout_ingc', type=float, default=0.3, help='动态关系学习dropout') #0.5

parser.add_argument('--eta', type=float, default=1, help='节点重要性') # 4
parser.add_argument('--gamma', type=float, default=0.0001, help='图的稀疏性') # 0.0001
parser.add_argument('--order', type=float, default=1, help='损失中 图损失占的比例')# 1
parser.add_argument('--temp', type=float, default=0.5, help='温度系数')

parser.add_argument('--layers', type=int, default=3, help='number of layers')

parser.add_argument('--column_wise', type=bool, default=False)
parser.add_argument('--test_ratio', type=float, default=0.2)
parser.add_argument('--val_ratio', type=float, default=0.2)
parser.add_argument('--lag', type=int, default=168, help='input time windows length')
parser.add_argument('--horizon', type=int, default=3, help='')
parser.add_argument('--output_len', type=int, default=1)
parser.add_argument('--headdim', type=int, default=8, help='每个头的维度')
parser.add_argument('--dilation_exponential_', type=int, default=5)

parser.add_argument('--MLP_layer', type=int, default=3) # 3
parser.add_argument('--MLP_dim', type=int, default=16) # 16 32
parser.add_argument('--if_node', type=bool, default=True)
parser.add_argument('--if_T_i_D', type=bool, default=True)
parser.add_argument('--if_D_i_W', type=bool, default=True)

parser.add_argument('--s_period', type=int, default=24)
parser.add_argument('--b_period', type=int, default=7)
parser.add_argument('--time_norm', type=bool, default=True)
parser.add_argument('--MLP_indim', type=int, default=3, help='inputs dimension of MLP')

parser.add_argument('--steps_per_day', type=int, default=24)

parser.add_argument('--embed_dim', type=int, default=10, help='node dim')
parser.add_argument('--temperature', type=float, default=0.5, help='')

parser.add_argument('--add_time_in_day', type=int, default=1)
parser.add_argument('--add_day_in_week', type=int, default=1)

parser.add_argument('--snorm_bool', type=int, default=1, help='')
parser.add_argument('--tnorm_bool', type=int, default=1, help='')
parser.add_argument('--use_RevIN', type=int, default=1, help='')

parser.add_argument('--init_temperature', type=float, default=2)
parser.add_argument('--temperature_decay', type=float, default=0.96)
parser.add_argument('--mini_temperature', type=float, default=0.05)

args = parser.parse_args()
log = open(args.log_file, 'w')


def log_string(string, log=log):
    log.write(string + '\n')
    log.flush()
    print(string)


def main():
    # set seed
    args.seed = int(time.time())
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    is_train = True # False #
    # load data
    device = torch.device(args.device)

    data = DataLoaderS_stamp(args.dataset, train=0.6, valid=0.2, device=device,
                       horizon=args.horizon, window=args.lag, normalize=args.normalize, args=args)

    print('loda dataset done')

    log_string(str(args))

    engine = trainer(data, distribute_data=None, args=args)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []

    if is_train:
        for i in range(1, args.epochs + 1):
            t1 = time.time()
            train_total_loss = 0
            n_samples = 0
            for iter, (x, y) in enumerate(data.get_batches(data.train[0], data.train[1], batch_size=args.batch_size)):
                trainx = x[..., :1]
                trainx = trainx.transpose(1, 3)
                trainy = y
                metrics = engine.train(trainx, trainy, data=data, pred_time_embed=x, iter=iter)
                train_total_loss += metrics
                n_samples += (y.size(0) * data.m)
                if iter % args.print_every == 0:
                    log = 'Iter: {:03d}, {:.4f} :Train Loss'
                    log_string(log.format(iter, metrics / (y.size(0) * data.m)))

            t2 = time.time()
            train_time.append(t2 - t1)
            # validation

            valid_total_loss = 0
            valid_total_loss_l1 = 0
            valid_n_samples = 0
            valid_output = None
            valid_label = None
            s1 = time.time()
            for iter, (x, y) in enumerate(data.get_batches(data.valid[0], data.valid[1], batch_size=args.batch_size)):
                trainx = x[..., :1]
                trainx = trainx.transpose(1, 3)
                trainy = y

                metrics = engine.eval(trainx, trainy, data, pred_time_embed=x)
                valid_total_loss += metrics[1]
                valid_total_loss_l1 += metrics[0]
                valid_n_samples += metrics[2]
                if valid_output is None:
                    valid_output = metrics[3]
                    valid_label = y
                else:
                    valid_output = torch.cat((valid_output, metrics[3]))
                    valid_label = torch.cat((valid_label, y))

            valid_rse, valid_rae, valid_correlation = generate_metric(valid_total_loss, valid_total_loss_l1,
                                                    valid_n_samples, data, valid_output, valid_label)

            engine.scheduler.step(valid_rse)

            s2 = time.time()
            val_time.append(s2 - s1)
            mtrain_loss = train_total_loss / n_samples

            mvalid_rse = valid_rse
            mvalid_rae = valid_rae
            mvalid_corr = valid_correlation
            his_loss.append(valid_rse.item())

            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Valid Loss: {:.4f}, ' \
                  'Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
            log_string(log.format(i, mtrain_loss,
                                  mvalid_rse, mvalid_rae, mvalid_corr, (t2 - t1)))

            torch.save(engine.model.state_dict(),
                       args.save + "_epoch_" + str(i) + ".pth")

        log_string("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        log_string("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

        # testing
        realy = []
        for x, y in data.get_batches(data.test[0], data.test[1], batch_size=args.batch_size):
            scale = data.scale.expand(y.size(0), data.m)
            realy.append(y * scale)

        bestid = np.argmin(his_loss)
        engine.model.load_state_dict(
            torch.load(args.save + "_epoch_" + str(bestid + 1) + ".pth"))
        engine.model.eval()
        outputs_r = []
        test_total_loss = 0
        test_total_loss_l1 = 0
        test_n_samples = 0
        test_predict = None
        test = None
        evaluatel2 = nn.MSELoss(size_average=False).to(device)
        evaluatel1 = nn.L1Loss(size_average=False).to(device)
        for iter, (x, y) in enumerate(data.get_batches(data.test[0], data.test[1], batch_size=args.batch_size)):
            testx = x[..., :1]
            testx = testx.transpose(1, 3)
            with torch.no_grad():
                preds, _, _, = engine.model(testx, pred_time_embed=x)
            preds = preds.squeeze()
            scale = data.scale.expand(preds.size(0), data.m)
            preds = preds * scale
            outputs_r.append(preds)

            test_total_loss += evaluatel2(preds, y * scale).item()
            test_total_loss_l1 += evaluatel1(preds, y * scale).item()
            test_n_samples += (preds.size(0) * data.m)

            if test_predict is None:
                test_predict = preds
                test = y
            else:
                test_predict = torch.cat((test_predict, preds))
                test = torch.cat((test, y))

        rse, rae, correlation = generate_metric(test_total_loss, test_total_loss_l1,
                                                test_n_samples, data, test_predict, test)

        log_string("The valid loss on best model is {}".format(str(round(his_loss[bestid], 4))))
        log_string('seed is {}'.format(args.seed))

        log = 'Evaluate best model on test data, Test rse: {:.4f}, Test rae: {:.4f}, Test corr: {:.4f}'
        log_string(log.format(rse, rae, correlation))
        torch.save(engine.model.state_dict(),
                   args.save + "_exp" + str(args.expid) + "_best_" + str(args.seed) + ".pth")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
