from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam

from models.select_network import define_G, define_M
from models.model_base import ModelBase
# from models.loss import CharbonnierLoss
# from models.loss_ssim import SSIMLoss

from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip


class ModelPlain(ModelBase):
    """Train with pixel loss"""

    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']  # training option
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)

        self.netM = define_M(opt)
        self.netM = self.model_to_device(self.netM)
        self.pool = torch.nn.AvgPool2d(kernel_size=3)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()  # load model
        self.netM.train()
        self.netG.train()  # set training mode,for BN
        # if want to load the pretrained model
        """

        self.netG.module.conv_first = nn.Conv2d(self.opt['n_channels'], self.opt['netG']['embed_dim'], 3, 1, 1)
        self.netG.module.conv_last = nn.Conv2d(64, self.opt['n_channels'], 3, 1, 1)
        self.netG = self.netG.to(self.device)                           # load model
        """
        self.define_loss()  # define loss
        self.define_optimizer()  # define optimizer
        self.load_optimizers()  # load optimizer
        self.define_scheduler()  # define scheduler
        self.log_dict = OrderedDict()  # log

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_M = self.opt['path']['pretrained_netM']
        if load_path_M is not None:
            print('Loading model for M [{:s}] ...'.format(load_path_M))
            self.load_network(load_path_M, self.netM, strict=self.opt_train['M_param_strict'], param_key='params')

        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'],
                                  param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerM = self.opt['path']['pretrained_optimizerM']
        if load_path_optimizerM is not None and self.opt_train['M_optimizer_reuse']:
            print('Loading optimizerM [{:s}] ...'.format(load_path_optimizerM))
            self.load_optimizer(load_path_optimizerM, self.M_optimizer)

        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netM, 'M', iter_label)
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)
        if self.opt_train['M_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.M_optimizer, 'optimizerM', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']
        self.M_lossfn_weight = self.opt_train['M_lossfn_weight']

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        if self.opt_train['G_optimizer_type'] == 'adam':
            self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'],
                                    betas=self.opt_train['G_optimizer_betas'],
                                    weight_decay=self.opt_train['G_optimizer_wd'])
        else:
            raise NotImplementedError
        self.M_optimizer = Adam(self.netM.parameters(), lr=self.opt_train['M_optimizer_lr'], weight_decay=0)

        M_optim_params = []
        for k, v in self.netM.named_parameters():
            if v.requires_grad:
                M_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        if self.opt_train['M_optimizer_type'] == 'adam':
            self.M_optimizer = Adam(M_optim_params, lr=self.opt_train['M_optimizer_lr'],
                                    betas=self.opt_train['M_optimizer_betas'],
                                    weight_decay=self.opt_train['M_optimizer_wd'])
        else:
            raise NotImplementedError

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        if self.opt_train['G_scheduler_type'] == 'MultiStepLR':
            self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                            self.opt_train['G_scheduler_milestones'],
                                                            self.opt_train['G_scheduler_gamma']
                                                            ))
        elif self.opt_train['G_scheduler_type'] == 'CosineAnnealingWarmRestarts':
            self.schedulers.append(lr_scheduler.CosineAnnealingWarmRestarts(self.G_optimizer,
                                                                            self.opt_train['G_scheduler_periods'],
                                                                            self.opt_train[
                                                                                'G_scheduler_restart_weights'],
                                                                            self.opt_train['G_scheduler_eta_min']
                                                                            ))
        else:
            raise NotImplementedError

        if self.opt_train['M_scheduler_type'] == 'MultiStepLR':
            self.schedulers.append(lr_scheduler.MultiStepLR(self.M_optimizer,
                                                            self.opt_train['M_scheduler_milestones'],
                                                            self.opt_train['M_scheduler_gamma']
                                                            ))
        elif self.opt_train['M_scheduler_type'] == 'CosineAnnealingWarmRestarts':
            self.schedulers.append(lr_scheduler.CosineAnnealingWarmRestarts(self.M_optimizer,
                                                                            self.opt_train['M_scheduler_periods'],
                                                                            self.opt_train[
                                                                                'M_scheduler_restart_weights'],
                                                                            self.opt_train['M_scheduler_eta_min']
                                                                            ))
        else:
            raise NotImplementedError

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        if need_H:
            self.H = data['H'].to(self.device)
        self.mask_label = torch.nan_to_num(self.pool(self.H) / self.L, nan=1.0)

        # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netM_forward(self):
        self.mask = self.netM(self.L) #self.mask_label 

    def netG_forward(self):
        self.E, self.supervised_nodes = self.netG(self.L, self.mask)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.M_optimizer.zero_grad()
        self.netM_forward()

        self.G_optimizer.zero_grad()
        self.netG_forward()
        G_loss = self.G_lossfn_weight * self.G_lossfn(self.E , self.H )
        #G_loss = self.G_lossfn_weight * self.G_lossfn(self.E*self.supervised_nodes, self.H*self.supervised_nodes)
        M_loss = self.M_lossfn_weight * self.G_lossfn(self.mask, self.mask_label)
        total_loss = G_loss + M_loss
        total_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        M_optimizer_clipgrad = self.opt_train['M_optimizer_clipgrad'] if self.opt_train['M_optimizer_clipgrad'] else 0
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0

        if M_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['M_optimizer_clipgrad'],
                                           norm_type=2)
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'],
                                           norm_type=2)

        self.M_optimizer.step()
        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        M_regularizer_orthstep = self.opt_train['M_regularizer_orthstep'] if self.opt_train[
            'M_regularizer_orthstep'] else 0
        if M_regularizer_orthstep > 0 and current_step % M_regularizer_orthstep == 0 and current_step % \
                self.opt['train']['checkpoint_save'] != 0:
            self.netM.apply(regularizer_orth)
        M_regularizer_clipstep = self.opt_train['M_regularizer_clipstep'] if self.opt_train[
            'M_regularizer_clipstep'] else 0
        if M_regularizer_clipstep > 0 and current_step % M_regularizer_clipstep == 0 and current_step % \
                self.opt['train']['checkpoint_save'] != 0:
            self.netM.apply(regularizer_clip)

        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train[
            'G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % \
                self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train[
            'G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % \
                self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        self.log_dict['G_loss'] = G_loss.item() / self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['G_loss'] = G_loss.item()
        self.log_dict['mask_loss'] = M_loss.item()

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netM.eval()
        self.netG.eval()
        with torch.no_grad():
            self.netM_forward()
        with torch.no_grad():
            self.netG_forward()
        self.netM.train()
        self.netG.train()

    # ----------------------------------------
    # test / inference x8
    # ----------------------------------------
    def testx8(self):
        self.netM.eval()
        self.netG.eval()
        with torch.no_grad():
            self.mask = test_mode(self.netM, self.L, mode=0, sf=self.opt['scale'], modulo=1)
            self.E = test_mode(self.netG, self.L, mask=self.mask, mode=5, sf=self.opt['scale'], modulo=1)
        self.netM.train()
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        out_dict['mask_e'] = self.mask.detach()[0].float().cpu()
        out_dict['mask_g'] = (self.mask_label).detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)
        msg = self.describe_network(self.netM)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)
        msg = self.describe_params(self.netM)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        msg += self.describe_network(self.netM)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        msg += self.describe_network(self.netM)
        return msg