import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from EMAmodel.loss import *
from EMAmodel.warplayer import warp
from EMAmodel.config import *

import torch.nn as nn
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from model.IFNet import *
import torch.nn.functional as F
from model.loss import *
from model.laplacian import *
from model.refine import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RIFE_Model:
    def __init__(self):
        self.flownet = IFNet()
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-3) # use large weight decay may avoid NaN loss
        self.epe = EPE()
        self.lap = LapLoss()
        self.sobel = SOBEL()

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
            torch.save(self.flownet.state_dict(),'{}.pkl'.format(path))

    def inference(self, img0, img1, scale=1, scale_list=[4, 2, 1], timestep=0.5):
        for i in range(3):
            scale_list[i] = scale_list[i] * 1.0 / scale
        imgs = torch.cat((img0, img1), 1)
        # Get all outputs including flow
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list, timestep=timestep)
        # Extract the final flow (corresponding to merged[2])
        final_flow = flow[2]
        # Return the final image and the final flow
        return merged[2], final_flow
    
    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt), 1), scale=[4, 2, 1])
        loss_l1 = (self.lap(merged[2], gt)).mean()
        loss_tea = (self.lap(merged_teacher, gt)).mean()
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_tea + loss_distill * 0.01 # when training RIFEm, the weight of loss_distill should be 0.005 or 0.002
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[2]
        return merged[2], {
            'merged_tea': merged_teacher,
            'mask': mask,
            'mask_tea': mask,
            'flow': flow[2][:, :2],
            'flow_tea': flow_teacher,
            'loss_l1': loss_l1,
            'loss_tea': loss_tea,
            'loss_distill': loss_distill,
            }


    def update_flow_distill(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None, teacher_model=None, flow_distill_weight=0.01, timestep=0.5): # Added timestep
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt), 1), scale=[4, 2, 1], timestep=timestep)
        loss_l1 = (self.lap(merged[2], gt)).mean()
        loss_tea = (self.lap(merged_teacher, gt)).mean() # RIFE's internal teacher loss
        # --- Flow Distillation ---
        loss_flow_distill = torch.tensor(0.0, device=imgs.device)
        if training and teacher_model is not None:
            with torch.no_grad(): 
                _, teacher_flow_target = teacher_model.inference_with_flow(img0, img1, timestep=timestep) # shape [B, 4, H, W]
            loss_flow_distill = (flow[2] - teacher_flow_target.detach()).abs().mean()
        # --- End Flow Distillation ---
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_tea + loss_distill * 0.01 + loss_flow_distill * flow_distill_weight
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[2]
        return merged[2], {
            'merged_tea': merged_teacher,
            'mask': mask,
            'mask_tea': mask,
            'flow': flow[2][:, :2],
            'flow_tea': flow_teacher,
            'loss_l1': loss_l1,
            'loss_tea': loss_tea,
            'loss_distill': loss_distill,
            'loss_flow_distill': loss_flow_distill
            }

    def update_output_distill(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None, teacher_model=None, ema_distill_weight=0.01):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt), 1), scale=[4, 2, 1])
        loss_l1 = (self.lap(merged[2], gt)).mean()
        loss_tea = (self.lap(merged_teacher, gt)).mean() # RIFE's internal teacher loss
        # --- EMA Teacher Distillation ---
        loss_ema_distill = 0
        if training and teacher_model is not None:
            with torch.no_grad():
                teacher_model.eval()
                ema_pred = teacher_model.inference(img0, img1)
            loss_ema_distill = (self.lap(merged[2], ema_pred.detach())).mean()
        # --- End EMA Teacher Distillation ---
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_tea + loss_distill * 0.01 + loss_ema_distill * ema_distill_weight
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[2]
        return merged[2], {
            'merged_tea': merged_teacher,
            'mask': mask,
            'mask_tea': mask, 
            'flow': flow[2][:, :2],
            'flow_tea': flow_teacher, 
            'loss_l1': loss_l1,
            'loss_tea': loss_tea, 
            'loss_distill': loss_distill, 
            'loss_ema_distill': loss_ema_distill 
            }

    
class EMA_VFI_Model:
    def __init__(self):
        backbonetype, multiscaletype = MODEL_CONFIG['MODEL_TYPE']
        backbonecfg, multiscalecfg = MODEL_CONFIG['MODEL_ARCH']
        self.net = multiscaletype(backbonetype(**backbonecfg), **multiscalecfg)
        self.name = MODEL_CONFIG['LOGNAME']
        self.device()
        self.optimG = AdamW(self.net.parameters(), lr=2e-4, weight_decay=1e-4)
        self.lap = LapLoss()

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def device(self):
        self.net.to(device)

    def load_model(self, name=None, rank=0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k and 'attn_mask' not in k and 'HW' not in k
            }
        if rank <= 0 :
            if name is None:
                name = self.name
            self.net.load_state_dict(convert(torch.load(f'ckpt/{name}.pkl')))
    
    def save_model(self, rank=0):
        if rank == 0:
            torch.save(self.net.state_dict(),f'ckpt/{self.name}.pkl')

    @torch.no_grad()
    def hr_inference(self, img0, img1, down_scale = 1.0, timestep = 0.5):
        '''
        Infer with down_scale flow
        Noting: return BxCxHxW
        '''
        def infer(imgs):
            img0, img1 = imgs[:, :3], imgs[:, 3:6]
            imgs_down = F.interpolate(imgs, scale_factor=down_scale, mode="bilinear", align_corners=False)
            flow, mask = self.net.calculate_flow(imgs_down, timestep)
            flow = F.interpolate(flow, scale_factor = 1/down_scale, mode="bilinear", align_corners=False) * (1/down_scale)
            mask = F.interpolate(mask, scale_factor = 1/down_scale, mode="bilinear", align_corners=False)
            af, _ = self.net.feature_bone(img0, img1)
            pred = self.net.coraseWarp_and_Refine(imgs, af, flow, mask)
            return pred
        imgs = torch.cat((img0, img1), 1)
        return infer(imgs)

    @torch.no_grad()
    def inference(self, img0, img1, timestep = 0.5):
        imgs = torch.cat((img0, img1), 1)
        _, _, _, pred = self.net(imgs, timestep=timestep)
        return pred

    @torch.no_grad()
    def inference_with_flow(self, img0, img1, timestep=0.5):
        """
        Performs inference and returns both the predicted frame and the estimated flow.
        """
        imgs = torch.cat((img0, img1), 1)
        # self.net(...) returns flow_list, mask_list, merged, pred
        flow_list, _, _, pred = self.net(imgs, timestep=timestep)
        # Get the final flow (last element in the list)
        final_flow = flow_list[-1]
        return pred, final_flow

    @torch.no_grad()
    def multi_inference(self, img0, img1, down_scale = 1.0, time_list=[]):
        '''
        Run backbone once, get multi frames at different timesteps
        Noting: return a list of [CxHxW]
        '''
        assert len(time_list) > 0, 'Time_list should not be empty!'
        def infer(imgs):
            img0, img1 = imgs[:, :3], imgs[:, 3:6]
            af, mf = self.net.feature_bone(img0, img1)
            imgs_down = None
            if down_scale != 1.0:
                imgs_down = F.interpolate(imgs, scale_factor=down_scale, mode="bilinear", align_corners=False)
                afd, mfd = self.net.feature_bone(imgs_down[:, :3], imgs_down[:, 3:6])
            pred_list = []
            for timestep in time_list:
                if imgs_down is None:
                    flow, mask = self.net.calculate_flow(imgs, timestep, af, mf)
                else:
                    flow, mask = self.net.calculate_flow(imgs_down, timestep, afd, mfd)
                    flow = F.interpolate(flow, scale_factor = 1/down_scale, mode="bilinear", align_corners=False) * (1/down_scale)
                    mask = F.interpolate(mask, scale_factor = 1/down_scale, mode="bilinear", align_corners=False)
                pred = self.net.coraseWarp_and_Refine(imgs, af, flow, mask)
                pred_list.append(pred)
            return pred_list
        imgs = torch.cat((img0, img1), 1)
        preds = infer(imgs)
        return [preds[i][0] for i in range(len(time_list))]
    
    def update(self, imgs, gt, learning_rate=0, training=True):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
            flow, mask, merged, pred = self.net(imgs)
            loss_l1 = (self.lap(pred, gt)).mean()
            for merge in merged:
                loss_l1 += (self.lap(merge, gt)).mean() * 0.5
            self.optimG.zero_grad()
            loss_l1.backward()
            self.optimG.step()
            return pred, loss_l1
        else: 
            self.eval()
            with torch.no_grad():
                flow, mask, merged, pred = self.net(imgs)
                return pred, 0