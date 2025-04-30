import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from model.IFNet import *
from model.IFNet_m import *
import torch.nn.functional as F
from model.loss import *
from model.laplacian import *
from model.refine import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class Model:
    def __init__(self, local_rank=-1, arbitrary=False):
        if arbitrary == True:
            self.flownet = IFNet_m()
        else:
            self.flownet = IFNet()
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-3) # use large weight decay may avoid NaN loss
        self.epe = EPE()
        self.lap = LapLoss()
        self.sobel = SOBEL()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
            
        if rank <= 0:
            self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))))
        
    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(),'{}/flownet.pkl'.format(path))

    def inference(self, img0, img1, scale=1, scale_list=[4, 2, 1], TTA=False, timestep=0.5):
        for i in range(3):
            scale_list[i] = scale_list[i] * 1.0 / scale
        imgs = torch.cat((img0, img1), 1)
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list, timestep=timestep)
        if TTA == False:
            return merged[2]
        else:
            flow2, mask2, merged2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3), scale_list, timestep=timestep)
            return (merged[2] + merged2[2].flip(2).flip(3)) / 2
    
    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None, teacher_model=None, ema_distill_weight=0.01):
        """
        Update the RIFE model.

        Args:
            imgs (Tensor): Input images (img0, img1 concatenated).
            gt (Tensor): Ground truth intermediate frame.
            learning_rate (float): Learning rate for the optimizer.
            mul (int): Multiplier (seems unused in the provided snippet).
            training (bool): Whether the model is in training mode.
            flow_gt (Tensor, optional): Ground truth flow (seems unused).
            teacher_model (nn.Module, optional): The pre-trained teacher model (EMA-VFI) for distillation.
            ema_distill_weight (float): Weight for the distillation loss from the EMA teacher.
        """
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()

        # RIFE forward pass (Student)
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt), 1), scale=[4, 2, 1])

        # Original RIFE losses
        loss_l1 = (self.lap(merged[2], gt)).mean()
        loss_tea = (self.lap(merged_teacher, gt)).mean() # RIFE's internal teacher

        # --- EMA Teacher Distillation ---
        loss_ema_distill = 0
        if training and teacher_model is not None:
            with torch.no_grad(): # Ensure no gradients are computed for the teacher
                teacher_model.eval() # Set teacher to evaluation mode
                # Assuming teacher_model has an 'inference' method similar to Trainer.py
                # Use the base inference without TTA for distillation target
                ema_pred = teacher_model.inference(img0, img1, TTA=False)
            # Calculate distillation loss (e.g., Laplacian loss between student and EMA teacher)
            loss_ema_distill = (self.lap(merged[2], ema_pred.detach())).mean()
        # --- End EMA Teacher Distillation ---

        if training:
            self.optimG.zero_grad()
            # Combine original losses with the new EMA distillation loss
            # Adjust weights as needed
            loss_G = loss_l1 + loss_tea + loss_distill * 0.01 + loss_ema_distill * ema_distill_weight
            loss_G.backward()
            self.optimG.step()
        else:
            # Keep RIFE's internal flow_teacher logic for evaluation if needed
            flow_teacher = flow[2] # Or potentially keep the original flow_teacher if used elsewhere

        return merged[2], {
            'merged_tea': merged_teacher,
            'mask': mask,
            'mask_tea': mask, # Note: Original code uses student mask for tea mask
            'flow': flow[2][:, :2],
            'flow_tea': flow_teacher, # RIFE's internal teacher flow
            'loss_l1': loss_l1,
            'loss_tea': loss_tea, # RIFE's internal teacher loss
            'loss_distill': loss_distill, # RIFE's internal distillation loss
            'loss_ema_distill': loss_ema_distill # Distillation loss from EMA model
            }
