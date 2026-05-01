from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import matplotlib as mpl
import matplotlib.cm as cm
from datetime import datetime
import json
import math
import string

# Custom models
from models.stereoanywhere import StereoAnywhere as StereoAnywhere

# Monocular models
from models.depth_anything_v2 import get_depth_anything_v2

from losses import *

from tensorboardX import SummaryWriter
import tqdm
import signal
import os

from dataloaders import fetch_dataloader
from utils import backup_source_code, normalize, estimate_normals, correlation_score

from torch.amp import GradScaler


class DummyResource:
    def __init__(self) -> None:
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

sigStopExecute = False
val_best = 10000000

parser = argparse.ArgumentParser(description='StereoAnywhere Training')

parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity for stereo model')
parser.add_argument('--model', default='stereoanywhere', help='select stereo model')
parser.add_argument('--loadmodel', default=None, help='start from pretrained model (insert path)')
parser.add_argument('--savemodel', default='./', help='instert the path where the model will be saved')
parser.add_argument('--iters', type=int, default=22, help='Number of iterations for recurrent networks')
parser.add_argument('--freeze_for_finetuning', action='store_true', help='Freeze some layers for finetuning')
parser.add_argument('--things_to_freeze', nargs='+', default=['fnet'], help='Layers to freeze for finetuning')
parser.add_argument('--init_disparity_zero', action='store_true', help='Initialize disparity to zero, instead of scaled mono')
parser.add_argument('--n_additional_hourglass', type=int, default=0, help='Number of additional hourglass modules')
parser.add_argument('--volume_channels', type=int, default=8, help='Number of channels in the volume')
parser.add_argument('--vol_n_masks', type=int, default=8, help='Number of masks in the volume')
parser.add_argument('--vol_downsample', type=float, default=0, help='Downsample factor for the volume')
parser.add_argument('--vol_aug_n_masks', type=int, default=4, help='Number of masks in the volume for augmentations')
parser.add_argument('--use_truncate_vol', action='store_true', help='Use the handcrafted truncation for stereo volume')
parser.add_argument('--mirror_conf_th', type=float, default=0.98, help='Confidence threshold for the truncation algorithm')
parser.add_argument('--mirror_attenuation', type=float, default=0.9, help='Attenuation factor for the truncation algorithm')
parser.add_argument('--use_aggregate_stereo_vol', action='store_true', help='Add hourglass layers to the stereo volume')
parser.add_argument('--use_aggregate_mono_vol', action='store_true', help='Add hourglass layers to the mono volume')
parser.add_argument('--normal_gain', type=int, default=10, help='Gain for normal estimation')
parser.add_argument('--lrc_th', type=float, default=1.0, help='Tthreshold for the left-right consistency')
parser.add_argument('--n_downsample', type=int, default=2, help='Number of downsampling layers for feature extraction')

parser.add_argument('--gt_mono_prob', type=float, default=0.3, help='Probability of using ground truth instead of mono prediction')
parser.add_argument('--volume_corruption_prob', type=float, default=0.3, help='Probability of corrupting the volumes')
parser.add_argument('--use_normal_loss', action='store_true', help='Regularize with normal loss')
parser.add_argument('--use_normal_loss_on_coarse', action='store_true', help='Regularize with normal loss on coarse predictions')
parser.add_argument('--use_border_mask', action='store_true', help='Mask the out-of-frame regions for the loss')

parser.add_argument('--monomodel', default='DAv2', help='Select monocular model')
parser.add_argument('--loadmonomodel', default=None, help='insert the path to the pretrained monocular model')

parser.add_argument('--dataset', default='sceneflow', help='choose the training dataset')
parser.add_argument('--datapath', default='', help='insert the path to the training dataset')
parser.add_argument('--datasetval', default='sceneflow', help='choose the validation dataset')
parser.add_argument('--datapathval', default='', help='insert the path to the validation dataset')
parser.add_argument('--overfit', action='store_true', default=False, help='select only one image for training')
parser.add_argument('--resume', action='store_true', help='Resume training from an intermediate checkpoint')
parser.add_argument('--numworkers', type=int, default=1, help='Number of workers for dataloader')

parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--maxsteps', type=int, default=0, help='number of max steps of training')
parser.add_argument('--batch_size', type=int, default=1, help='size of single batches')
parser.add_argument('--image_size', type=int, nargs='+', default=[384, 384], help='image patch size for training')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--initstep', type=int, default=0, help='Start the training from step <initstep>')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='Set random seed')


parser.add_argument('--savestep', type=int, default=1, help='Save intermediate model every <savestep> epochs')
parser.add_argument('--plotstep', type=int, default=200, help='Plot loss every <plotstep> steps')
parser.add_argument('--imagestep', type=int, default=1000, help='Plot a debug image every <imagestep> steps')
parser.add_argument('--valstep', type=int, default=5000, help='Make an evaluation every <valstep> steps')

parser.add_argument('--do_validation', action='store_true', help='Enable validation')
parser.add_argument('--valsize', type=int, default=0, help='Validation max size (0=unlimited)')
parser.add_argument('--valmetric', default='rmse', choices=['rmse', 'bad1', 'bad2', 'bad3', 'mae'], help='Validation metric to use (it will select the best model)')
parser.add_argument('--iscale', type=int, default=1, help='Downsampling factor for input images (Only for validation)')
parser.add_argument('--oscale', type=int, default=1, help='Downsampling factor for output disparity (Only for validation)')

parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
parser.add_argument('--debug_grad', action='store_true', help='Debug gradients')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--preload_mono', action='store_true')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.debug_grad:
    dict_of_inf_params = {}
    dict_of_nan_params = {}

torch.autograd.set_detect_anomaly(False)


def random_string(n=8):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))


MYRND_STRING = random_string(3)

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda')
    autocast_device = "cuda"
else:
    device = torch.device('cpu')
    autocast_device = "cpu"

scaler = GradScaler(autocast_device, enabled=args.mixed_precision)

if args.model == 'stereoanywhere':
    model = StereoAnywhere(args)
else:
    print('no model')
    exit()

if args.loadmodel is not None and args.freeze_for_finetuning:
    model.freeze_for_finetuning()

if args.cuda:
    model = nn.DataParallel(model)
    model = model.to(device)

if args.monomodel == 'DAv2':
    mono_model = get_depth_anything_v2(args.loadmonomodel, map_location=device)
    mono_model = mono_model.to(device)
else:
    mono_model = None

pretrain_dict = None

if args.loadmodel is not None:
    print('Load pretrained model')
    state_dict = 'state_dict'

    if args.model in ['stereoanywhere']:
        pretrain_dict = torch.load(args.loadmodel, map_location=device)
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model.load_state_dict(pretrain_dict, strict=False)
    else:
        pretrain_dict = torch.load(args.loadmodel, map_location=device)
        model.load_state_dict(pretrain_dict[state_dict])
else:
    print('No pretrained model')

n_params = sum([p.data.nelement() for p in model.parameters()])
print('Number of model parameters: {}'.format(n_params))

args.test = False
train_loader = fetch_dataloader(args)

tmp_dataset = args.dataset
tmp_datapath = args.datapath
tmp_batchsize = args.batch_size
tmp_numworkers = args.numworkers

args.dataset = args.datasetval
args.datapath = args.datapathval
args.batch_size = 1
args.numworkers = 0
args.test = True
val_loader = fetch_dataloader(args)

args.dataset = tmp_dataset
args.datapath = tmp_datapath
args.batch_size = tmp_batchsize
args.numworkers = tmp_numworkers

if args.model in ['stereoanywhere']:
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=.00001, eps=1e-8 if not args.mixed_precision else 1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.epochs*len(train_loader)+100, pct_start=0.001, cycle_momentum=False, anneal_strategy='linear')
else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

# Assume scheduler and optimizer associated with stereo netowork .tar
if args.loadmodel is not None and args.resume:
    pretrain_dict = torch.load(args.loadmodel, map_location=device)

    if optimizer is not None and "optimizer" in pretrain_dict:
        print("Loading Optimizer state")
        optimizer.load_state_dict(pretrain_dict['optimizer'])

    if scheduler is not None and "scheduler" in pretrain_dict:
        print("Loading Scheduler state")
        scheduler.load_state_dict(pretrain_dict['scheduler'])
        print(f"Scheduler learning rate: {scheduler.get_last_lr()}")

    if "current_step" in pretrain_dict:
        args.initstep = pretrain_dict['current_step']+1

print(f"Learning rate: {args.lr}")

H, W = args.image_size
_, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
xx = xx.to(device).reshape(1, 1, H, W).float()


def train(data):
    model.train()
    mono_model.eval()

    has_gt_right = 'gt_right' in data

    if args.model in ['stereoanywhere']:
        model.module.freeze_bn()

    for k in ['im2', 'im3', 'im2_aug', 'im3_aug', 'gt', 'validgt', 'gt_right', 'validgt_right']:
        if k in data:
            data[k] = data[k].to(device)

    if args.model not in ['stereoanywhere']:
        data['im2_mono'], data['im3_mono'] = torch.zeros_like(data['gt']), torch.zeros_like(data['gt'])
    elif random.random() < args.gt_mono_prob and has_gt_right:
        _gt = torch.clone(data['gt'])
        _gt[torch.isinf(_gt)] = 0
        _gt[torch.isnan(_gt)] = 0
        _gtr = torch.clone(data['gt_right'])
        _gtr[torch.isinf(_gtr)] = 0
        _gtr[torch.isnan(_gtr)] = 0
        data['im2_mono'], data['im3_mono'] = _gt, _gtr
    elif ('im2_mono' not in data or 'im3_mono' not in data):
        with torch.no_grad():
            with torch.autocast(autocast_device, enabled=args.mixed_precision):
                if args.monomodel == 'DAv2':
                    data['im2_mono'] = mono_model.infer_image(data['im2'], input_size_width=518, input_size_height=518)
                    data['im3_mono'] = mono_model.infer_image(data['im3'], input_size_width=518, input_size_height=518)

    data['im2_mono'], data['im3_mono'] = normalize([data['im2_mono'].detach().float().to(device), data['im3_mono'].detach().float().to(device)])

    optimizer.zero_grad()

    with torch.autograd.set_detect_anomaly(False):

        # Use mixed-precision if enabled
        with torch.autocast(autocast_device, enabled=args.mixed_precision):
            if args.model in ['stereoanywhere']:
                pred_disps, pred_confs, disps0, disps1, confs0, confs1 = model(data['im2_aug'], data['im3_aug'], data['im2_mono'], data['im3_mono'], iters=args.iters, test_mode=False)
            else:
                pred_disps = model(data['im2_aug'], data['im3_aug'])

        if args.model in ['stereoanywhere']:
            mask = (data['validgt'] > 0) & (data['gt'] < args.maxdisp)
            conf_th = args.lrc_th
            div_const = math.log(1+math.exp(conf_th))
            normal_gain_loss = 10
            iterative_gain_loss = 1
            _normals_im2_mono = estimate_normals(data['im2_mono'], normal_gain=data['im2_mono'].shape[-1]/args.normal_gain)
            _normals_im3_mono = estimate_normals(data['im3_mono'], normal_gain=data['im3_mono'].shape[-1]/args.normal_gain)

            loss_gamma = 0.9
            n_predictions = len(pred_disps)
            all_losses = []
            for i in range(n_predictions):
                # We adjust the loss_gamma so it is consistent for any number of iterations
                adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
                i_weight = adjusted_loss_gamma**(n_predictions - i - 1)

                _tmp = (pred_disps[i][mask] - -data['gt'][mask]).abs()
                all_losses.append(i_weight * iterative_gain_loss * _tmp.mean())

                if args.model in ['stereoanywhere']:
                    if args.use_normal_loss:
                        _normals_a = estimate_normals(normalize(-pred_disps[i])[0], normal_gain=data['im2_mono'].shape[-1]/args.normal_gain)
                        _normals_loss = (1-correlation_score(_normals_a, _normals_im2_mono))[mask].mean()
                        if not torch.isnan(_normals_loss):
                            all_losses.append(i_weight * iterative_gain_loss * normal_gain_loss * _normals_loss)

                    if pred_confs[i] is not None:
                        _tmp = (pred_disps[i][mask] - data['gt'][mask]).abs()
                        conf_gt = F.softplus(conf_th-(_tmp)).detach().float() / div_const
                        nan_mask = torch.logical_not(torch.isnan(conf_gt)) & torch.logical_not(torch.isnan(pred_confs[i][mask]))
                        _clip_a = torch.clip(pred_confs[i][mask][nan_mask], 0, 1)
                        _clip_b = torch.clip(conf_gt[nan_mask], 0, 1)
                        _binary_loss = F.binary_cross_entropy(_clip_a, _clip_b)
                        if not torch.isnan(_binary_loss):
                            all_losses.append(i_weight * iterative_gain_loss * _binary_loss)

            if args.model in ['stereoanywhere']:
                left_border_mask = xx-data['gt'] >= 0 if args.use_border_mask else torch.ones_like(data['gt']).bool()

                for i, (disp0, conf0) in enumerate(zip(disps0, confs0)):
                    if disp0 is not None and not torch.isnan(disp0).any():
                        if i == 2:  # train the scaler on full frame
                            _tmp = (disp0[mask] - data['gt'][mask]).abs()
                        else:
                            _tmp = (disp0[mask & left_border_mask] - data['gt'][mask & left_border_mask]).abs()
                            if args.use_normal_loss_on_coarse:
                                _normals_a = estimate_normals(normalize(disp0)[0], normal_gain=data['im2_mono'].shape[-1]/args.normal_gain)
                                _normals_loss = (1-correlation_score(_normals_a, _normals_im2_mono))[mask & left_border_mask].mean()
                                if not torch.isnan(_normals_loss):
                                    all_losses.append(i_weight * normal_gain_loss * _normals_loss)

                        _tmp_loss = _tmp.mean()
                        if not torch.isnan(_tmp_loss):
                            all_losses.append(_tmp_loss)

                        if conf0 is not None:
                            _tmp = (disp0[mask] - data['gt'][mask]).abs()
                            conf0_gt = F.softplus(conf_th-(_tmp)).detach().float() / div_const
                            nan_mask = torch.logical_not(torch.isnan(conf0_gt)) & torch.logical_not(torch.isnan(conf0[mask]))
                            _clip_a = torch.clip(conf0[mask][nan_mask], 0, 1)
                            _clip_b = torch.clip(conf0_gt[nan_mask], 0, 1)
                            _binary_loss = F.binary_cross_entropy(_clip_a, _clip_b)
                            all_losses.append(_binary_loss)

                if has_gt_right:
                    mask_right = (data['validgt_right']>0) & (data['gt_right'] < args.maxdisp)
                    right_border_mask = xx+data['gt_right'] < W if args.use_border_mask else torch.ones_like(data['gt_right']).bool()                    

                    for i, (disp1, conf1) in enumerate(zip(disps1, confs1)):
                        if disp1 is not None and not torch.isnan(disp1).any():

                            if i == 2:  # train the scaler on full frame
                                _tmp = (disp1[mask_right] - data['gt_right'][mask_right]).abs()
                            else:
                                _tmp = (disp1[mask_right & right_border_mask] - data['gt_right'][mask_right & right_border_mask]).abs()
                                if args.use_normal_loss_on_coarse:
                                    _normals_a = estimate_normals(normalize(disp1)[0], normal_gain=data['im3_mono'].shape[-1]/args.normal_gain)
                                    _normals_loss = (1-correlation_score(_normals_a, _normals_im3_mono))[
                                        mask_right & right_border_mask].mean()
                                    if not torch.isnan(_normals_loss):
                                        all_losses.append(i_weight * normal_gain_loss * _normals_loss)

                            _tmp_loss = _tmp.mean()
                            if not torch.isnan(_tmp_loss):
                                all_losses.append(_tmp_loss)

                            if conf1 is not None:
                                _tmp = (disp1[mask_right] - data['gt_right'][mask_right]).abs()
                                conf1_gt = F.softplus(conf_th-(_tmp)).detach().float() / div_const
                                nan_mask = torch.logical_not(torch.isnan(conf1_gt)) & torch.logical_not(torch.isnan(conf1[mask_right]))
                                _clip_a = torch.clip(conf1[mask_right][nan_mask], 0, 1)
                                _clip_b = torch.clip(conf1_gt[nan_mask], 0, 1)

                                _binary_loss = F.binary_cross_entropy(_clip_a, _clip_b)
                                if not torch.isnan(_binary_loss):
                                    all_losses.append(_binary_loss)

            loss = sum(all_losses)
            pred_disp = -pred_disps[-1].squeeze(1)

        if not torch.isnan(loss) and not torch.isinf(loss):
            scaler.scale(loss).backward()

    loss = loss.detach()

    valid_gradients = True
    found_nan = False
    found_inf = False

    if args.debug_grad:
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    found_nan = torch.isnan(param.grad).any()
                    found_inf = torch.isinf(param.grad).any()

                    # valid_gradients = not found_nan # not (found_nan or found_inf)
                    valid_gradients = True

                    # NaNGrad to 0 instead of skip
                    # inf values are clipped next
                    param.grad = torch.nan_to_num_(param.grad, nan=0.0)

                    if found_nan:
                        dict_of_nan_params[param_name] = dict_of_nan_params.get(param_name, 0) + 1

                    if found_inf:
                        dict_of_inf_params[param_name] = dict_of_inf_params.get(param_name, 0) + 1

                    if not valid_gradients:
                        break

        # Save dict of nan and inf params to json
        with open(os.path.join(args.savemodel, 'nan_params.json'), 'w') as f:
            json.dump(dict_of_nan_params, f)
        with open(os.path.join(args.savemodel, 'inf_params.json'), 'w') as f:
            json.dump(dict_of_inf_params, f)

    if not valid_gradients and args.debug_grad:
        print(f'[{datetime.now()}] detected inf ({found_inf}) or nan ({found_nan}) values in gradients. not updating model parameters')
    else:
        if args.model in ['stereoanywhere']:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        if args.model in ['stereoanywhere']:
            scheduler.step()

    return {'loss': loss.data, 'pred_disp': F.interpolate(pred_disp.data.unsqueeze(1), size=(data['gt'].shape[-2], data['gt'].shape[-1]), mode='bilinear', align_corners=False).squeeze(1) * (data['gt'].shape[-1] / pred_disp.shape[-1])}


@torch.no_grad()
def val(data):
    model.eval()
    model_tmp = model.module if isinstance(model, nn.DataParallel) else model

    if args.monomodel in ['DAv2', 'depthpro'] and mono_model is not None:
        mono_model.eval()

    if args.iscale != 1:
        data['im2'] = F.interpolate(data['im2'], scale_factor=1./args.iscale)
        data['im3'] = F.interpolate(data['im3'], scale_factor=1./args.iscale)

    if args.oscale != 1:
        data['gt'] = F.interpolate(data['gt'], scale_factor=1./args.oscale, mode='nearest') / args.oscale
        data['validgt'] = F.interpolate(data['validgt'], scale_factor=1./args.oscale, mode='nearest')

    data['im2'], data['im3'] = data['im2'].to(device), data['im3'].to(device)

    if args.monomodel == 'DAv2':
        data['im2_mono'] = mono_model.infer_image(data['im2'], input_size_width=518, input_size_height=518)
        data['im2_mono'] = (data['im2_mono'] - data['im2_mono'].min()) / (data['im2_mono'].max() - data['im2_mono'].min())
        data['im3_mono'] = mono_model.infer_image(data['im3'], input_size_width=518, input_size_height=518)
        data['im3_mono'] = (data['im3_mono'] - data['im3_mono'].min()) / (data['im3_mono'].max() - data['im3_mono'].min())
    else:
        data['im2_mono'], data['im3_mono'] = torch.zeros_like(data['gt']), torch.zeros_like(data['gt'])

    ht, wt = data['im2'].shape[-2], data['im2'].shape[-1]

    pad_ht = (((ht // 32) + 1) * 32 - ht) % 32
    pad_wd = (((wt // 32) + 1) * 32 - wt) % 32

    _pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
    data['im2'] = F.pad(data['im2'], _pad, mode='replicate')
    data['im3'] = F.pad(data['im3'], _pad, mode='replicate')
    data['im2_mono'] = F.pad(data['im2_mono'], _pad, mode='replicate')
    data['im3_mono'] = F.pad(data['im3_mono'], _pad, mode='replicate')

    with torch.autocast(enabled=args.mixed_precision, device_type=autocast_device, dtype=torch.float16):
        if args.model in ['stereoanywhere']:
            pred_disps, _ = model_tmp(data['im2'], data['im3'], data['im2_mono'], data['im3_mono'], test_mode=True, iters=32)
        else:
            pred_disps = model_tmp(data['im2'], data['im3'])

    if args.model in ['stereoanywhere']:
        pred_disp = -pred_disps.squeeze(1)
    else:
        raise NotImplementedError

    ht, wd = pred_disp.shape[-2:]
    c = [_pad[2], ht-_pad[3], _pad[0], wd-_pad[1]]
    pred_disp = pred_disp[..., c[0]:c[1], c[2]:c[3]]

    if args.iscale != 1 and args.iscale/args.oscale != 1:
        pred_disp = F.interpolate(pred_disp.unsqueeze(0), scale_factor=args.iscale/args.oscale, mode='nearest').squeeze(0) * args.iscale / args.oscale

    result = guided_metrics(pred_disp.cpu().numpy(), data['gt'].numpy(), data['validgt'].numpy())

    return result


def colormap_image(x, vmax=None, cmap='Spectral_r'):
    """Apply colormap to image pixels
    """
    ma = float(x[0].cpu().data.max()) if vmax is None else vmax
    mi = float(0)
    normalizer = mpl.colors.Normalize(vmin=mi, vmax=ma)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    colormapped_im = (mapper.to_rgba(x[0].cpu())[:, :, :3] * 255).astype(np.uint8)
    return np.transpose(colormapped_im, (2, 0, 1))


def signal_handler(signum, frame):
    global sigStopExecute
    signal_name = {signal.SIGUSR1: "SIGUSR1", signal.SIGTERM: "SIGTERM"}
    print(f"{signal_name[signum]} Received! Stopping training")
    sigStopExecute = True


def main():
    signal.signal(signal.SIGUSR1, signal_handler)

    step = 0
    start_full_time = time.time()
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S_%f")
    args.savemodel = os.path.join(os.path.dirname(
        args.savemodel), f"{args.model}_{dt_string}_{MYRND_STRING}")

    writer = SummaryWriter(os.path.join(args.savemodel, "log"))

    with open(os.path.join(args.savemodel, "args.json"), 'w') as args_json:
        json.dump(args.__dict__, args_json, indent=4)

    backup_source_code(os.path.join(args.savemodel, "code"))

    total_steps = args.epochs * len(train_loader)
    maxsteps = args.maxsteps if args.maxsteps > 0 else total_steps
    total_steps = min(maxsteps, total_steps)
    pbar = tqdm.tqdm(total=total_steps)

    try:
        for epoch in range(1, args.epochs+1):
            if epoch*len(train_loader) < args.initstep:
                pbar.set_description('Epoch %d SKIPPING EPOCH.....' % (epoch), False)
                pbar.update(len(train_loader))
                step += len(train_loader)
                continue

            if step >= maxsteps:
                break

            fast_skipping = True
            if fast_skipping:
                while step < args.initstep:
                    pbar.set_description('Epoch %d FAST SKIPPING STEP.....' % (epoch), False)
                    pbar.update(1)
                    step += 1

            ## training ##
            for datablob in train_loader:
                if step < args.initstep:
                    pbar.set_description('Epoch %d SKIPPING STEP.....' % (epoch), False)
                    pbar.update(1)
                    step += 1
                    continue

                if step >= maxsteps:
                    break

                result = train(datablob)
                if step % args.plotstep == 0:
                    writer.add_scalar("{}".format("loss"), result['loss'], step)

                if step % args.imagestep == 0:
                    vald = datablob['gt'][0][datablob['validgt'][0] > 0]

                    row0 = np.concatenate((datablob['im2_aug'].data[0].cpu(), datablob['im3_aug'].data[0].cpu()), 2)
                    rowb = np.concatenate((datablob['im2_aug'].data[0].cpu(), datablob['im3_aug'].data[0].cpu()), 2)

                    d21 = colormap_image(datablob['gt'][0], vmax=vald.max())/255.*(datablob['validgt'][0, 0].cpu().numpy() > 0).astype(np.float32)
                    row1 = np.concatenate((d21, colormap_image(result['pred_disp'], vmax=vald.max())/255.), 2)

                    writer.add_image("cat_pred_disp/{}".format(0), np.concatenate((row0, rowb, row1), 1), step)
                    writer.flush()

                pbar.set_description('Epoch %d, loss %.2f, lr %.8f' % (epoch, result['loss'], optimizer.param_groups[0]['lr']), False)

                if (step) % args.valstep == 0 and args.do_validation:
                    del datablob
                    del result
                    torch.cuda.empty_cache()

                    bad1_list = []
                    bad2_list = []
                    bad3_list = []
                    rms_list = []
                    mae_list = []
                    val_len = min(len(val_loader), args.valsize) if args.valsize > 0 else len(val_loader)
                    for val_step, val_datablob in enumerate(val_loader):
                        result = val(val_datablob)
                        bad1_list.append(result['bad 1.0'])
                        bad2_list.append(result['bad 2.0'])
                        bad3_list.append(result['bad 3.0'])
                        mae_list.append(result['avgerr'])
                        rms_list.append(result['rms'])
                        pbar.set_description(f"Validation {val_step+1}/{len(val_loader)} ({100*result['bad 3.0']:.2f}%) ...")

                        if val_step >= val_len:
                            break

                    mean_result1 = 100*(np.array(bad1_list).mean())
                    mean_result2 = 100*(np.array(bad2_list).mean())
                    mean_result3 = 100*(np.array(bad3_list).mean())
                    mean_result4 = np.array(rms_list).mean()
                    mean_result5 = np.array(mae_list).mean()

                    if args.valmetric == 'bad1':
                        val_metric = mean_result1
                    elif args.valmetric == 'bad2':
                        val_metric = mean_result2
                    elif args.valmetric == 'bad3':
                        val_metric = mean_result3
                    elif args.valmetric == 'rmse':
                        val_metric = mean_result4
                    else:
                        val_metric = mean_result5

                    writer.add_scalar("{}".format("validation_bad1"), mean_result1, step)
                    writer.add_scalar("{}".format("validation_bad2"), mean_result2, step)
                    writer.add_scalar("{}".format("validation_bad3"), mean_result3, step)
                    writer.add_scalar("{}".format("validation_rmse"), mean_result4, step)
                    writer.add_scalar("{}".format("validation_mae"), mean_result5, step)

                    pbar.set_description(f"Validation completed ({mean_result1:.2f}%,{mean_result2:.2f}%,{mean_result3:.2f}%,{mean_result4:.2f} RMS,{mean_result5:.2f} MAE) ...")

                    global val_best
                    if val_metric < val_best:
                        val_best = val_metric
                        savefilename = args.savemodel+'/checkpoint_best.tar'
                        torch.save({'state_dict': model.state_dict()}, savefilename)

                    del val_datablob
                    del result
                    torch.cuda.empty_cache()

                step += 1
                pbar.update(1)

                if sigStopExecute:
                    raise KeyboardInterrupt

            if epoch % args.savestep == 0:
                savefilename = args.savemodel+'/checkpoint_'+str(epoch)+'.tar'
                state_dict = {'state_dict': model.state_dict()}
                state_dict['optimizer'] = optimizer.state_dict()
                state_dict['scheduler'] = scheduler.state_dict()
                state_dict['current_step'] = step
                torch.save(state_dict, savefilename)

    except KeyboardInterrupt:
        print('Stopping and saving')
        savefilename = args.savemodel+'/checkpoint_stopped.tar'
        state_dict = {'state_dict': model.state_dict()}
        state_dict['optimizer'] = optimizer.state_dict()
        state_dict['scheduler'] = scheduler.state_dict()
        state_dict['current_step'] = step
        torch.save(state_dict, savefilename)

        writer.close()
        exit()

    writer.close()

    print('full training time = %.2f HR' %
          ((time.time() - start_full_time)/3600))

    savefilename = args.savemodel+'/checkpoint_'+str(epoch)+'.tar'
    torch.save({'state_dict': model.state_dict()}, savefilename)


if __name__ == '__main__':
    main()
