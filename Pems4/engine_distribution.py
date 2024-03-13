import torch.optim as optim
from .model_distribution import *
from .util_distribution import *
from torch.optim import lr_scheduler
import torch
from .lib.metrics import RMSE_torch, MAE_torch, MAPE_torch


class trainer():
    def __init__(self, scaler, distribute_data, args):
        self.model = TPGCN(args=args, distribute_data=distribute_data)
        self.model.to(args.device)

        nparams = sum([p.nelement() for p in self.model.parameters()])
        print(nparams)

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                                        patience=20, eps=0.00001, cooldown=30, verbose=True)

        self.loss = MAE_torch # masked_mae
        self.scaler = scaler
        self.clip = 1
        self.loss_usual = nn.SmoothL1Loss() # nn.MSELoss() #

    def train(self, input, real_val, iter=0, pred_time_embed=None):
        accumulation_steps = 1
        self.model.train()

        output, gl_loss, MLP_out = self.model(input, pred_time_embed)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        # real = torch.unsqueeze(real_val, dim=1)
        real = self.scaler.inverse_transform(real_val)
        predict = self.scaler.inverse_transform(output)

        if gl_loss is None:
            loss = self.loss_usual(predict, real)
        else:
            loss = self.loss_usual(predict, real)

        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        if ((iter + 1) % accumulation_steps) == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        mape = MAPE_torch(predict, real, 0.0).item() # masked_mape
        rmse = RMSE_torch(predict, real, 0.0).item() # masked_rmse
        return loss.item(), mape, rmse

    def eval(self, input, real_val, pred_time_embed=None):
        self.model.eval()
        output, _, _ = self.model(input, pred_time_embed=pred_time_embed)
        output = output.transpose(1, 3)
        real = self.scaler.inverse_transform(real_val)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = MAPE_torch(predict, real, 0.0).item() # masked_mape
        rmse = RMSE_torch(predict, real, 0.0).item() # masked_rmse
        return loss.item(), mape, rmse
