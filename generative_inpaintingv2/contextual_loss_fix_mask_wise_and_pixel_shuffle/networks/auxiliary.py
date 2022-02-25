import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import gen_conv1 , gen_conv2, gen_deconv1 ,gen_deconv1_origin ,  gen_deconv2,gen_deconv2_origin, dis_conv
import torch.optim as optim
from .utils import batch_conv2d, batch_transposeconv2d, weight_init ,create_dir,imsave
from .utils import CP1, CP2 , ICNR
import pdb
import os
from .dataset import Dataset
from torch.utils.data import DataLoader
from .convnet import InpaintGenerator ,InpaintGeneratorOrigin ,InpaintDiscriminator , InpaintDiscriminator_change
from .nearestx2 import InpaintGeneratorLarge , InpaintGeneratorHighRes
from .convnet_origin import InpaintGeneratorBase
from tensorboardX import SummaryWriter



from .utils import Progbar, create_dir, stitch_images, imsave

torch.autograd.set_detect_anomaly(True)


class ResumableRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    #data_source: Sized
    #replacement: bool

    def __init__(self, data_source):
        self.data_source = data_source
        self.generator = torch.Generator()
        self.generator.manual_seed(10)
        
        self.perm_index = 0
        self.perm = torch.randperm(self.num_samples, generator=self.generator)
        
    @property
    def num_samples(self) -> int:
        return len(self.data_source)

    def __iter__(self):
        if self.perm_index >= len(self.perm):
            self.perm_index = 0
            self.perm = torch.randperm(self.num_samples, generator=self.generator)
            
        while self.perm_index < len(self.perm):
            self.perm_index += 1
            yield self.perm[self.perm_index-1]

    def __len__(self):
        return self.num_samples
    
    def get_state(self):
        return {"perm": self.perm, "perm_index": self.perm_index, "generator_state": self.generator.get_state()}
    
    def set_state(self, state):
        self.perm = state["perm"]
        self.perm_index = state["perm_index"]
        self.generator.set_state(state["generator_state"])

class AdversarialLoss(nn.Module):
    """
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='hinge', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss






class InpaintGeneratorFull(nn.Module):
    """auxiliary network.
    """
    def __init__(self, config,cnum=48, nn_hard=False, baseg=None, rate=1):
        super(InpaintGeneratorFull, self).__init__()
        self.config = config
        self.name = "InpaintGenerator"
        self.gen_full_weights_path = os.path.join(config.PATH, self.name + '_gen_full.pth')
        self.gen_base_weights_path = os.path.join(config.PATH, self.name + '_gen_base.pth')
        # self.dis_weights_path = os.path.join(config.PATH, self.name + '_dis.pth')
        self.trainbase=False
        self.baseg = baseg
        # self.based = based
        self.iteration = 0
        # self.cnum = cnum
        # if len(config.GPU) > 1:
        #     self.baseg = nn.DataParallel(self.baseg, config.GPU)
        #     self.based = nn.DataParallel(self.based, config.GPU)
        # self.adversarial_loss = AdversarialLoss()
        # if self.config.MODE == 2:
        #     self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_EDGE_FLIST, config.TEST_MASK_FLIST, augment=False, training=False)
        # else:
        #     self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_EDGE_FLIST, config.TRAIN_MASK_FLIST, augment=True, training=True)
        #     self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_EDGE_FLIST, config.VAL_MASK_FLIST, augment=False, training=True)
        #     self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        # self.samples_path = os.path.join(config.PATH, 'samples')
        # self.results_path = os.path.join(config.PATH, 'results')




        # similarity encoder
        self.sconv1 = gen_conv2(2*cnum, 4*cnum, 5, 1) # skip cnn out
        self.sconv2 = gen_conv2(2*cnum, 4*cnum, 3, 1, activation=nn.ReLU()) # skip cnn out

        # auxiliary encoder
        self.bconv1 = gen_conv2(3, cnum, 5, 1) # skip cnn out
        self.bconv2_downsample = gen_conv2(int(cnum/2), 2*cnum, 3, 2)
        self.bconv3 = gen_conv2(cnum, 2*cnum, 3, 1) # skip cnn out
        self.bconv4_downsample = gen_conv2(cnum, 4*cnum, 3, 2)

        self.conv13_upsample_conv = gen_deconv2(2*cnum, 2*cnum)
        self.conv14 = gen_conv2(cnum*2, 2*cnum, 3, 1) # skip cnn in
        self.conv15_upsample_conv = gen_deconv2(cnum, cnum)
        self.conv16 = gen_conv2(cnum, cnum, 3, 1) # skip cnn in
        # auxiliary decoder
        self.conv16_2 = gen_conv2(cnum//2, cnum, 3, 1)
        self.conv17 = gen_conv2(cnum//2, 3, 3, 1, activation=None)
        if config.DIS_CHANGE:
          self.cp_1 = CP1(nn_hard=nn_hard, ufstride=2*rate, 
                stride=2*rate, bkg_patch_size=4*rate, pd=1*rate,is_fuse = True)
        else:
          self.cp_1 = CP1(nn_hard=nn_hard, ufstride=2*rate, 
                stride=2*rate, bkg_patch_size=4*rate, pd=1*rate)
        self.cp_2 = CP2(ufstride=2*4*rate, bkg_patch_size=16*rate, 
                stride=8*rate, pd=4*rate)
        self.apply(weight_init)

    # def get_param_list(self, stage="all"):
    #     if stage=="all":
    #         list_param = [p for name, p in self.named_parameters()]
    #         return list_param
    #     elif stage=="base":
    #         list_param = [p for name, p in self.baseg.named_parameters()]
    #         return list_param
    #     else:
    #         raise NotImplementedError

    def save(self):
        torch.save({
            'iteration': self.iteration,
            # 'generator_base': self.baseg.state_dict(),
            'generator_full':self.state_dict()
        }, self.gen_full_weights_path)
        torch.save({'generator_base': self.baseg.state_dict()},self.gen_base_weights_path)

        # torch.save({
        #     'discriminator': self.based.state_dict()
        # }, self.dis_weights_path)

        
        # self.baseg.save()
    def load(self):
        if os.path.exists(self.gen_full_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_full_weights_path)
            else:
                data = torch.load(self.gen_full_weights_path, map_location=lambda storage, loc: storage)

            self.load_state_dict(data['generator_full'])
            self.iteration = data['iteration']
            # kernel = ICNR(self.baseg.conv13_upsample_conv.weight)
            # self.baseg.conv13_upsample_conv.weight.data.copy_(kernel)
            # print("Load 1")
            # kernel = ICNR(self.baseg.conv15_upsample_conv.weight)
            # self.baseg.conv15_upsample_conv.weight.data.copy_(kernel)
            # print("Load 2")
            # kernel = ICNR(self.baseg.allconv13_upsample_conv.weight)
            # self.baseg.allconv13_upsample_conv.weight.data.copy_(kernel)
            # print("Load 3")
            # kernel = ICNR(self.baseg.allconv15_upsample_conv.weight)
            # self.baseg.allconv15_upsample_conv.weight.data.copy_(kernel)
            # print("Load 4")
            # kernel = ICNR(self.conv13_upsample_conv.weight)
            # self.conv13_upsample_conv.weight.data.copy_(kernel)
            # print("Load 5")
            # kernel = ICNR(self.conv15_upsample_conv.weight)
            # self.conv15_upsample_conv.weight.data.copy_(kernel)
            # print("Load 6")

    def forward(self, x, mask):
        self.iteration += 1
        _,_,hin,win = x.shape
        x_stage1, x_stage2, pm = self.baseg(x, mask)
        # if (not self.training) or self.trainbase:
        #     print("aaaa")
        #     return x_stage1, x_stage2, x_stage2

        # similarify
        xnow = x_stage2*mask + x*(1-mask)
        xs = self.sconv1(pm)
        x_similar = self.sconv2(xs)

        bsize, _, h, w = xs.size()
        mask_s = F.avg_pool2d(mask, kernel_size=4, stride=4)
        similar = self.cp_1(x_similar, x_similar, mask_s)

        xb = self.bconv1(xnow)
        x_skip1 = xb
        xb = self.bconv2_downsample(xb)
        xb = self.bconv3(xb)
        x_skip2 = xb
        xb = self.bconv4_downsample(xb)
        xb = self.conv13_upsample_conv(xb)
        xb = self.conv14(torch.cat((xb, x_skip2), 1))
        xb = self.conv15_upsample_conv(xb)
        xb = self.conv16(torch.cat((xb, x_skip1), 1))

        xb = self.cp_2(similar, xb, mask)

        xb = self.conv16_2(xb)
        xb = self.conv17(xb)
        xb = torch.tanh(xb)
        return x_stage1, x_stage2, xb

class InpaintGeneratorFullOrigin(nn.Module):
    """auxiliary network.
    """
    def __init__(self, config,cnum=48, nn_hard=False, baseg=None, rate=1):
        super(InpaintGeneratorFullOrigin, self).__init__()
        self.config = config
        self.name = "InpaintGenerator"
        self.gen_full_weights_path = os.path.join(config.PATH, self.name + '_gen_full.pth')
        self.gen_base_weights_path = os.path.join(config.PATH, self.name + '_gen_base.pth')
        # self.dis_weights_path = os.path.join(config.PATH, self.name + '_dis.pth')
        self.trainbase=False
        self.baseg = baseg
        # self.based = based
        self.iteration = 0
        # self.cnum = cnum
        # if len(config.GPU) > 1:
        #     self.baseg = nn.DataParallel(self.baseg, config.GPU)
        #     self.based = nn.DataParallel(self.based, config.GPU)
        # self.adversarial_loss = AdversarialLoss()
        # if self.config.MODE == 2:
        #     self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_EDGE_FLIST, config.TEST_MASK_FLIST, augment=False, training=False)
        # else:
        #     self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_EDGE_FLIST, config.TRAIN_MASK_FLIST, augment=True, training=True)
        #     self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_EDGE_FLIST, config.VAL_MASK_FLIST, augment=False, training=True)
        #     self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        # self.samples_path = os.path.join(config.PATH, 'samples')
        # self.results_path = os.path.join(config.PATH, 'results')




        # similarity encoder
        self.sconv1 = gen_conv2(2*cnum, 4*cnum, 5, 1) # skip cnn out
        self.sconv2 = gen_conv2(2*cnum, 4*cnum, 3, 1, activation=nn.ReLU()) # skip cnn out

        # auxiliary encoder
        self.bconv1 = gen_conv2(3, cnum, 5, 1) # skip cnn out
        self.bconv2_downsample = gen_conv2(int(cnum/2), 2*cnum, 3, 2)
        self.bconv3 = gen_conv2(cnum, 2*cnum, 3, 1) # skip cnn out
        self.bconv4_downsample = gen_conv2(cnum, 4*cnum, 3, 2)

        self.conv13_upsample_conv = gen_deconv2_origin(2*cnum, 2*cnum)
        self.conv14 = gen_conv2(cnum*2, 2*cnum, 3, 1) # skip cnn in
        self.conv15_upsample_conv = gen_deconv2_origin(cnum, cnum)
        self.conv16 = gen_conv2(cnum, cnum, 3, 1) # skip cnn in
        # auxiliary decoder
        self.conv16_2 = gen_conv2(cnum//2, cnum, 3, 1)
        self.conv17 = gen_conv2(cnum//2, 3, 3, 1, activation=None)
        if config.DIS_CHANGE:
          self.cp_1 = CP1(nn_hard=nn_hard, ufstride=2*rate, 
                stride=2*rate, bkg_patch_size=4*rate, pd=1*rate,is_fuse = True)
        else:
          self.cp_1 = CP1(nn_hard=nn_hard, ufstride=2*rate, 
                stride=2*rate, bkg_patch_size=4*rate, pd=1*rate)
        self.cp_2 = CP2(ufstride=2*4*rate, bkg_patch_size=16*rate, 
                stride=8*rate, pd=4*rate)
        self.apply(weight_init)

    # def get_param_list(self, stage="all"):
    #     if stage=="all":
    #         list_param = [p for name, p in self.named_parameters()]
    #         return list_param
    #     elif stage=="base":
    #         list_param = [p for name, p in self.baseg.named_parameters()]
    #         return list_param
    #     else:
    #         raise NotImplementedError

    def save(self):
        torch.save({
            'iteration': self.iteration,
            # 'generator_base': self.baseg.state_dict(),
            'generator_full':self.state_dict()
        }, self.gen_full_weights_path)
        torch.save({'generator_base': self.baseg.state_dict()},self.gen_base_weights_path)

        # torch.save({
        #     'discriminator': self.based.state_dict()
        # }, self.dis_weights_path)

        
        # self.baseg.save()
    def load(self):
        if os.path.exists(self.gen_full_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_full_weights_path)
            else:
                data = torch.load(self.gen_full_weights_path, map_location=lambda storage, loc: storage)

            self.load_state_dict(data['generator_full'])
            self.iteration = data['iteration']

            

    def forward(self, x, mask):
        self.iteration += 1
        _,_,hin,win = x.shape
        x_stage1, x_stage2, pm = self.baseg(x, mask)
        # if (not self.training) or self.trainbase:
        #     print("aaaa")
        #     return x_stage1, x_stage2, x_stage2

        # similarify
        xnow = x_stage2*mask + x*(1-mask)
        xs = self.sconv1(pm)
        x_similar = self.sconv2(xs)

        bsize, _, h, w = xs.size()
        mask_s = F.avg_pool2d(mask, kernel_size=4, stride=4)
        similar = self.cp_1(x_similar, x_similar, mask_s)

        xb = self.bconv1(xnow)
        x_skip1 = xb
        xb = self.bconv2_downsample(xb)
        xb = self.bconv3(xb)
        x_skip2 = xb
        xb = self.bconv4_downsample(xb)
        xb = self.conv13_upsample_conv(xb)
        xb = self.conv14(torch.cat((xb, x_skip2), 1))
        xb = self.conv15_upsample_conv(xb)
        xb = self.conv16(torch.cat((xb, x_skip1), 1))

        xb = self.cp_2(similar, xb, mask)

        xb = self.conv16_2(xb)
        xb = self.conv17(xb)
        xb = torch.tanh(xb)
        return x_stage1, x_stage2, xb
    # def train(self):
    #     train_loader = DataLoader(
    #         dataset=self.train_dataset,
    #         batch_size=self.config.BATCH_SIZE,
    #         num_workers=4,
    #         drop_last=True,
    #         shuffle=True
    #     )

    #     epoch = 0
    #     keep_training = True
    #     model = self.config.MODEL
    #     max_iteration = int(float((self.config.MAX_ITERS)))
    #     total = len(self.train_dataset)
    #     for items in train_loader:
    #         images,  masks = self.cuda(*items)
    #         gen_loss = 0
    #         dis_loss = 0
    #         x1 ,x2,xb = self(images,masks)
    #         dis_input_real = images
    #         dis_input_fake = x2.detach()
    #         dis_real = self.based(dis_input_real)                    # in: [rgb(3)]
    #         dis_fake = self.based(dis_input_fake)                    # in: [rgb(3)]
    #         dis_real_loss = self.adversarial_loss(dis_real, True, True)
    #         dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
    #         dis_loss += (dis_real_loss + dis_fake_loss) / 2
    #         gen_input_fake = outputs
    #         gen_fake = self.based(gen_input_fake)                    # in: [rgb(3)]
    #         gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
    #         gen_loss += gen_gan_loss
    #         loss_l1 = self.config.L1_LOSS_WEIGHT *torch.nn.L1Loss()(x1,images) + self.config.L1_LOSS_WEIGHT*torch.nn.L1Loss()(x2,images)
    #         gen_loss += loss_l1



    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)
        

class InpaintModel(nn.Module):
    def __init__(self,config,base_g = None,cnum=48):
        super(InpaintModel, self).__init__()
        self.name = "InpaintModel"
        
        if base_g is True:
          self.flag = True
          self.baseg = InpaintGeneratorFullOrigin(config,baseg = InpaintGeneratorLarge()).to(config.DEVICE)
          print("a")
        elif base_g is False:
          print("b")
          self.baseg = InpaintGeneratorFull(config,baseg = InpaintGenerator()).to(config.DEVICE)
          self.flag = True
        elif base_g is None:
          print("c")
          self.flag = True
          self.baseg = InpaintGeneratorFullOrigin(config,baseg = InpaintGeneratorOrigin()).to(config.DEVICE)
        if config.DIS_CHANGE:
          self.based = InpaintDiscriminator_change().to(config.DEVICE)
        else:
          self.based = InpaintDiscriminator().to(config.DEVICE)
        self.writer = SummaryWriter(str(config.NAME))
        # self.iteration = 0
        self.config = config
        self.cnum = cnum
        self.dis_weights_path = os.path.join(config.PATH, "DiscriminatorModel" + '_dis.pth')
        if len(config.GPU) > 1:
            self.baseg = nn.DataParallel(self.baseg, config.GPU)
            self.based = nn.DataParallel(self.based, config.GPU)

        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_EDGE_FLIST, config.TEST_MASK_FLIST, augment=False, training=False)
        else:
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_EDGE_FLIST, config.TRAIN_MASK_FLIST, augment=True, training=True)
            self.resume = ResumableRandomSampler(self.train_dataset)
            self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_EDGE_FLIST, config.VAL_MASK_FLIST, augment=False, training=True)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        # self.samples_path = os.path.join(config.PATH, 'samples')
        # self.results_path = os.path.join(config.PATH, 'results')
        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True
        
        self.gen_optimizer = optim.Adam(
            params=self.baseg.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=self.based.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )
        self.resume_path = os.path.join(config.PATH, "resume_data.pth")
        # self.resume_path = os.path.join(config.PATH, "test.pth")
        self.opt_gen_path = os.path.join(config.PATH, "opt_gen_state.pth")
        self.opt_dis_path = os.path.join(config.PATH, "opt_dis_state.pth")
        self.l1_1 = torch.nn.L1Loss()
        self.l1_2 = torch.nn.L1Loss()
        # self.l1_2.register_backward_hook(self._backward_hook)
        self.l1_context = torch.nn.L1Loss()

        self.adversarial_loss = AdversarialLoss()
        self.adversarial_gen_loss = AdversarialLoss()
        # self.adversarial_gen_loss.register_backward_hook(self._backward_hook)
        # self.train_loader = None

    # def _backward_hook(self, module, grad_input, grad_output):
    #     # print(len(grad_input))
    #     print(len(grad_output))
    #     # print("grad_input",grad_input)
    #     print("grad_output",grad_output)

    def save(self):
        self.baseg.save()
        
        torch.save(self.resume.get_state(), self.resume_path)

        torch.save(self.gen_optimizer.state_dict(), self.opt_gen_path)
        torch.save(self.dis_optimizer.state_dict(), self.opt_dis_path)

        torch.save({
            'discriminator': self.based.state_dict()
        }, self.dis_weights_path)

    def load(self):
        self.baseg.load()
        
        if self.config.MODE == 1 and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)

            self.based.load_state_dict(data['discriminator'])
        if self.config.MODE == 1 and os.path.exists(self.resume_path):
          print('Loading resume data...' )

          if torch.cuda.is_available():
            self.resume.set_state(torch.load(self.resume_path))
          else:
            self.resume.set_state(torch.load(self.resume_path, map_location=lambda storage, loc: storage))
        
        if self.config.MODE == 1 and os.path.exists(self.opt_gen_path):
          print('Loading gen opt and dis opt state')
          if torch.cuda.is_available():
              data_opt_gen = torch.load(self.opt_gen_path)
              data_opt_dis = torch.load(self.opt_dis_path)
          else:
              data_opt_gen = torch.load(self.opt_gen_path, map_location=lambda storage, loc: storage)
              data_opt_dis = torch.load(self.opt_dis_path, map_location=lambda storage, loc: storage)
          
          self.gen_optimizer.load_state_dict(data_opt_gen)
          self.dis_optimizer.load_state_dict(data_opt_dis)
          
    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            sampler=self.resume,
            num_workers=4,
            drop_last=True,
            shuffle=False
        )

        epoch = 0
        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)
        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return
        self.baseg.train()
        self.based.train()
        print(len(self.train_dataset))
        print(len(train_loader))
        while(keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)
            if self.flag:
              progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])
            for  items in  train_loader:
                # print(name)

                self.gen_optimizer.zero_grad()
                self.dis_optimizer.zero_grad()
                images,  masks = self.cuda(*items)
                # print(masks.shape)
                gen_loss = 0
                dis_loss = 0
                context_loss = 0
                
                # real_images = images * 2. - 1
                real_images = (images-0.5)/0.5
                images = real_images*(1-masks)
                x1 ,x2,xb = self.baseg(images,masks)
                output = x2*masks + images*(1-masks)
                xb_output = xb*masks + images*(1-masks)
                dis_input_real = real_images
                dis_input_fake = output.detach()
                # dis_input_context = xb.detach()
                dis_real = self.based(dis_input_real,masks)                    # in: [rgb(3)]
                dis_fake = self.based(dis_input_fake,masks)
                # dis_context  = self.based(dis_input_context)          # in: [rgb(3)]
                dis_real_loss = self.adversarial_loss(dis_real, True, True)
                dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
                dis_loss += (dis_real_loss + dis_fake_loss) / 2

                # dis_loss.backward(retain_graph=True)
                # self.dis_optimizer.step()

                gen_input_fake = output
                gen_input_context = xb_output
                gen_fake = self.based(gen_input_fake,masks)  
                gen_context = self.based(gen_input_context,masks)                 # in: [rgb(3)]
                # gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
                gen_gan_loss = self.adversarial_gen_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
                gen_context_loss = self.adversarial_loss(gen_context, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
                context_loss += gen_context_loss
                gen_loss += gen_gan_loss

                l1_2 = self.config.L1_LOSS_WEIGHT*self.l1_2(x2,real_images)
                loss_l1 = self.config.L1_LOSS_WEIGHT *self.l1_1(x1,real_images) + l1_2
                
                loss_l1_context = self.l1_context(xb,real_images)
                context_loss += self.config.BETA*loss_l1_context
                ###
                # grad_gen_to_x2 = torch.autograd.grad(outputs =gen_loss, inputs =x2,retain_graph=True)
                # l1_to_x2 = torch.autograd.grad(outputs = loss_l1,inputs= x2,retain_graph=True)
                # print(len(grad_gen_to_x2))
                # print(torch.sum(torch.abs(grad_gen_to_x2[0])))
                # print(torch.sum(torch.abs(l1_to_x2[0])))

                # grad_gen_to_x2 = torch.autograd.grad(outputs =gen_loss, inputs =x2,retain_graph=True)
                # l1_to_x2 = torch.autograd.grad(outputs = loss_l1,inputs= x2,retain_graph=True)

                # print(torch.sum(torch.abs(grad_gen_to_x2[0])))
                # print(torch.sum(torch.abs(l1_to_x2[0])))
                
                # self.writer.add_scalar("grad_gen_to_x2",torch.autograd.grad(outputs =gen_loss, inputs =x2))
                # self.writer.add_scalar("l1_to_x2",torch.autograd.grad(outputs = loss_l1,inputs= x2))
                ###
                gen_loss +=  self.config.BETA*loss_l1
            
                gen_loss += self.config.LAMBDA*context_loss

                iteration = self.baseg.iteration
                if self.flag:
                  logs = [
                      ("epoch", epoch),
                      ("iter", iteration),
                      ("gen_loss",gen_loss.item()),
                      ("dis_loss",dis_loss.item()),
                      ("l1_loss",loss_l1.item()),
                      ("l1_loss_context",loss_l1_context.item()),
                      ("gen_context_loss",gen_context_loss.item()),
                      ("gen_gan_loss",gen_gan_loss.item()),
                      ("loss_l1_2",l1_2.item())
                  ]
                  # logs = [
                  #   ("grad_gen_to_x2",float(torch.sum(torch.abs(torch.autograd.grad(gen_loss, x2,retain_graph = True,only_inputs = True)[0])))),
                  #   ("l1_to_x2",float(torch.sum(torch.abs(torch.autograd.grad(loss_l1, x2,retain_graph = True,only_inputs = True)[0])))),
                  #   ("grad_gen_context_to_x2",float(torch.sum(torch.abs(torch.autograd.grad(context_loss, xb,retain_graph = True,only_inputs = True)[0])))),
                  #   ("l1_context_to_x2",float(torch.sum(torch.abs(torch.autograd.grad(loss_l1_context, xb,retain_graph = True,only_inputs = True)[0]))))
                  # ]

                  #     ("grad_gen_to_x2",torch.autograd.grad(gen_loss.detach(), x2.detach())),
                  #     ("grad_gen_to_xb",torch.autograd.grad(gen_loss.detach(), xb.detach()))
                  
                ########
                self.writer.add_scalar('dis_loss', dis_loss.item(),iteration)
                self.writer.add_scalar('gen_loss', gen_loss.item(),iteration)
                self.writer.add_scalar('l1_loss', loss_l1.item(),iteration)
                self.writer.add_scalar('l1_loss_context', loss_l1_context.item(),iteration)
                self.writer.add_scalar('gen_context_loss', gen_context_loss.item(),iteration)
                self.writer.add_scalar('gen_gan_loss', gen_gan_loss.item(),iteration)
              
                self.writer.add_scalar("grad_gen_to_x2",torch.sum(torch.abs(torch.autograd.grad(gen_loss, x2,retain_graph = True,only_inputs = True)[0])),iteration)
                self.writer.add_scalar("l1_to_x2",torch.sum(torch.abs(torch.autograd.grad(loss_l1, x2,retain_graph = True,only_inputs = True)[0])),iteration)
                #######
                # self.writer.add_scalar("grad_gen_to_x2",torch.autograd.grad(gen_loss.detach(), x2.detach()))
                # self.writer.add_scalar("l1_to_x2",torch.autograd.grad(loss_l1.detach(), x2.detach()))
                # self.writer.add_scalar('grad_gen_to_x2', gen_context_loss.item(),iteration)
                # self.writer.add_scalar('gen_gan_loss', gen_gan_loss.item(),iteration)
                # x2.retain_grad()


                dis_loss.backward()
                self.dis_optimizer.step()
                gen_loss.backward()
                self.gen_optimizer.step()
              


                if iteration >= max_iteration:
                    keep_training = False
                    self.writer.close()
                    break

                # logs = [
                #     ("epoch", epoch),
                #     ("iter", iteration),
                #     ("gen_loss",gen_loss.item()),
                #     ("dis_loss",dis_loss.item())
                # #     ("grad_gen_to_x2",torch.autograd.grad(gen_loss.detach(), x2.detach())),
                # #     ("grad_gen_to_xb",torch.autograd.grad(gen_loss.detach(), xb.detach()))
                # ]
                if self.flag:
                  progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                  self.save()
                  
                if self.config.SAMPLE_INTERVAL and iteration % (2*self.config.SAMPLE_INTERVAL) == 0:
                  self.sample()
                
                # if iteration % 10 == 0 :
                #   torch.save(self.resume.get_state(), self.resume_path)
                #   keep_training = False
                #   self.writer.close()
                #   break
                # self.writer.add_scalar('dis_loss', dis_loss.item(),iteration)
                # self.writer.add_scalar('gen_loss', gen_loss.item(),iteration)
                # self.writer.add_scalar('grad_gen_to_x2', torch.autograd.grad(gen_loss.detach(), x2.detach()),iteration)
                # self.writer.add_scalar('grad_gen_to_xb', torch.autograd.grad(gen_loss.detach(), xb.detach()),iteration)
    def test(self):
        self.baseg.eval()
        self.based.eval()
        # net = InpaintGeneratorHighRes()
        # net.load_state_dict(torch.load(os.path.join("/content/drive/MyDrive/generative_inpaintingv2/contextual_loss_fix_mask_wise_and_pixel_shuffle/files" ,'model_near512.pth')))

        # net = InpaintGeneratorBase()
        # net.load_state_dict(torch.load(os.path.join("/content/drive/MyDrive/generative_inpaintingv2/contextual_loss_fix_mask_wise_and_pixel_shuffle/files" ,'model_256.pth')))

        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        # count = 0
        for items in test_loader:
            name = self.test_dataset.load_name(index)
            images,  masks = self.cuda(*items)

            index += 1
            images = (images-0.5)/0.5
            inputs = images*(1-masks)
            x1 ,x2,xb = self.baseg(inputs,masks)
            

            outputs = x2
            outputs_merged = (outputs * masks) + (inputs * (1 - masks))

            
                  
            
            output = self.postprocess(outputs_merged)[0]
            path = os.path.join(self.results_path, name)
            print(index, name)
            imsave(output, path)
            # x1 ,x2 = net(inputs,masks)
            # outputs = x2
            # outputs_merged = (outputs * masks) + (inputs * (1 - masks))
            # output = self.postprocess(outputs_merged)[0]
            # imsave(output, os.path.join("/content/drive/MyDrive/Places2/validation_of_cr_origin",name))
            # imsave(output, os.path.join("/content/drive/MyDrive/Places2/validation_of_cr_origin_high_res",name))

            if index == 1000:
              break
          
    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        self.baseg.eval()
        # self.based.eval()

        # model = self.config.MODEL
        items = next(self.sample_iterator)
        # images, images_gray, edges, masks = self.cuda(*items)
        images,  masks = self.cuda(*items)
        # images = images *2. -1
        # edge model
    
        iteration = self.baseg.iteration
        # inputs = (images_gray * (1 - masks)) + masks
        # images = images *2. -1
        images = (images-0.5)/0.5
        inputs = images*(1-masks)
        # inputs = inputs *2. -1
        # outputs = self.edge_model(images_gray, edges, masks)
        x1 ,x2,xb = self.baseg(inputs,masks)
        outputs = x2
        outputs_merged = (outputs * masks) + (inputs * (1 - masks))

        # inpaint model
        # elif model == 2:
        #     iteration = self.inpaint_model.iteration
        #     inputs = (images * (1 - masks)) + masks
        #     outputs = self.inpaint_model(images, edges, masks)
        #     outputs_merged = (outputs * masks) + (images * (1 - masks))

        # # inpaint with edge model / joint model
        # else:
        #     iteration = self.inpaint_model.iteration
        #     inputs = (images * (1 - masks)) + masks
        #     outputs = self.edge_model(images_gray, edges, masks).detach()
        #     edges = (outputs * masks + edges * (1 - masks)).detach()
        #     outputs = self.inpaint_model(images, edges, masks)
        #     outputs_merged = (outputs * masks) + (images * (1 - masks))

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1

        images = stitch_images(
            self.postprocess(images),
            # self.postprocess(inputs),
            self.inputpostprocess(images,masks),
            self.postprocess(outputs),
            self.postprocess(outputs_merged),
            img_per_row = image_per_row
        )


        path = os.path.join(self.samples_path, "inpaintfull")
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = (img+1) * 127.5
        img = img.permute(0, 2, 3, 1)
        return img.int()
    
    def inputpostprocess(self,img,mask):
        img = (img+1) * 127.5
        img = img*(1-mask)
        img = img.permute(0, 2, 3, 1)
        return img.int()


    



    
    




# if __name__ == "__main__":
#     pass
