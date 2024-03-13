import torch.optim as optim
from .model_distribution import *
from .util import *
from torch.optim import lr_scheduler
import torch


class trainer():
    def __init__(self, data, distribute_data, args):

        self.model = gwnet(args=args, distribute_data=distribute_data)

        self.model.to(args.device)
        self.gc_order = args.order

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                                        patience=10, eps=0.00001, cooldown=20, verbose=True)

        self.loss = nn.L1Loss(size_average=False).to(args.device)
        self.loss_mse = nn.MSELoss(size_average=False).to(args.device)

        nparams = sum([p.nelement() for p in self.model.parameters()])
        print(nparams)
        self.clip = 0.5
        self.loss_usual = nn.SmoothL1Loss()

    def train(self, input, real_val, data, pred_time_embed=None, iter=0):
        accumulation_steps = 1
        self.model.train()
        output, gl_loss, _ = self.model(input, pred_time_embed)
        output = output.squeeze()

        real = real_val
        scale = data.scale.expand(real.size(0), data.m)
        real, predict = real * scale, output * scale

        if gl_loss is None:
            loss = self.loss(predict, real)
        else:
            loss = self.loss(predict, real) + torch.mean(gl_loss) * self.gc_order

        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        if ((iter + 1) % accumulation_steps) == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item()

    def eval(self, input, real_val, data, pred_time_embed=None):
        self.model.eval()
        with torch.no_grad():
            output, _, _, = self.model(input, pred_time_embed)
        output = output.squeeze()
        real = real_val
        scale = data.scale.expand(real.size(0), data.m)
        real, predict = real * scale, output * scale
        loss_mse = self.loss_mse(predict, real)
        loss = self.loss(predict, real)
        samples = (output.size(0) * data.m)
        return loss.item(), loss_mse.item(), samples, output
