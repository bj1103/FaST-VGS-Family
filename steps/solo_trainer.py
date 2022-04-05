import time
import os
import torch
import math
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datasets import spokencoco_dataset, places_dataset, flickr8k_dataset, libri_dataset
from datasets.sampler import StatefulSampler
from models import fast_vgs
from .utils import *
from .trainer_utils import *
from .bert_adam import BertAdam
from logging import getLogger
logger = getLogger(__name__)

class Trainer:
    @staticmethod
    def add_args(parser):
        parser.add_argument("--seed", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=2)
        parser.add_argument("--exp_dir", type=str)
        parser.add_argument("--trained_weights_dir", type=str, default=None)
        parser.add_argument("--batch_size", type=int)
        parser.add_argument("--val_batch_size", type=int)
        parser.add_argument("--val_cross_batch_size", type=int)
        parser.add_argument("--n_epochs", type=int)
        parser.add_argument("--n_print_steps", type=int)
        parser.add_argument("--n_val_steps", type=int)
        parser.add_argument("--lr", type=float, default=0.0001)
        parser.add_argument("--warmup_fraction", type=float, default=0.1)
        parser.add_argument("--solo_loss", type=str, default=None)
        parser.add_argument("--grad_accum", type=int, default=1)
        parser.add_argument("--projector_mlp", type=str, default="1536-1536", help="Size and number of layers of the MLP expander head")
        parser.add_argument("--same_projector", action="store_true", default=True, help="Same projector for audio and image")
        parser.add_argument("--sim_coeff_coarse", type=float, default=2.5, help="Coarse invariance regularization loss coefficient for VICReg")
        parser.add_argument("--std_coeff_coarse", type=float, default=2.5, help="Coarse variance regularization loss coefficient for VICReg")
        parser.add_argument("--cov_coeff_coarse", type=float, default=0.1, help="Coarse covariance regularization loss coefficient for VICReg")
        parser.add_argument("--sim_coeff_fine", type=float, default=25.0, help="Fine invariance regularization loss coefficient for VICReg")
        parser.add_argument("--std_coeff_fine", type=float, default=25.0, help="Fine variance regularization loss coefficient for VICReg")
        parser.add_argument("--cov_coeff_fine", type=float, default=1.0, help="Fine covariance regularization loss coefficient for VICReg")
        parser.add_argument("--lambd", type=float, default=0.0051, help="Weight on off-diagonal terms for barlow twins")
    
    def __init__(self, args):
        self.start_time = time.time()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"number of devices: {torch.cuda.device_count()}")
        self.writer = SummaryWriter(self.args.exp_dir)
        self.seed_everything(seed=self.args.seed)
        self.meters = self._setup_meters()
        self.progress, self.total_progress = setup_progress(self)
        
        if self.args.solo_loss == 'VICReg':
            self.solo_module_coarse = fast_vgs.VICReg(self.args, 'coarse').to(self.device)
            self.solo_module_fine = fast_vgs.VICReg(self.args, 'fine').to(self.device)
        elif self.args.solo_loss == 'BarlowTwins':
            self.solo_module_coarse = fast_vgs.BarlowTwins(self.args).to(self.device)
            self.solo_module_fine = fast_vgs.BarlowTwins(self.args).to(self.device)
           
        self.dual_encoder, self.cross_encoder, self.trainables, self.indices, self.libri_indices, self.optim_states = self._setup_models()
        self.use_libri_loss = self.args.libri_w2v2_weight != None
        self.train_loader, self.valid_loader, self.valid_loader2, self.train_sampler, self.libri_train_loader, self.libri_valid_loader, self.libri_train_sampler, self.train_data_length = self._setup_dataloader()
        self.total_num_updates = int(math.floor(self.train_data_length / self.args.batch_size))*self.args.n_epochs
        self.optimizer = self._setup_optimizer()
        if torch.cuda.device_count() > 1:
            self.dual_encoder = nn.DataParallel(self.dual_encoder)
            self.cross_encoder = nn.DataParallel(self.cross_encoder)
        self.scheduler = self._setup_scheduler()
        self.criterion = fast_vgs.Margin_InfoNCE_loss
        logger.info(f"batch size: {self.args.batch_size}")
        
         
    def forward(self, batch):
        audio_feats, audio_cls, extended_audio_attention_mask, visual_feats, visual_cls, losses = self.dual_encoder(audio_feats = batch['audio'], attention_mask = batch['audio_attention_mask'], visual_feats = batch['visual_feats'], visual_pos = batch['visual_pos'], target_list = batch['label'])
        coarse_cross_relationship_score_matrix = visual_cls @ audio_cls.transpose(0,1)
        losses['coarse_matching_loss'] = fast_vgs.Margin_InfoNCE_loss(coarse_cross_relationship_score_matrix, margin=self.args.margin, img_id = batch['img_id'])
        if self.args.fine_matching_weight != 0:
            B = visual_feats.shape[0]
            visual_feats_square = visual_feats.repeat(B,1,1)
            audio_feats_square = audio_feats.repeat_interleave(B, dim=0)
            extended_audio_attention_mask_square = extended_audio_attention_mask.repeat_interleave(B, dim=0)
            cross_relationship_score_square = self.cross_encoder(audio_feats_square, extended_audio_attention_mask_square, visual_feats_square)
            cross_relationship_score_matrix = cross_relationship_score_square.view(B,B)
            losses["fine_matching_loss"] = fast_vgs.Margin_InfoNCE_loss(cross_relationship_score_matrix, margin=self.args.margin, img_id = batch['img_id'])
        return losses

    def forward_solo(self, batch):
        audio_feats, audio_cls, extended_audio_attention_mask, visual_feats, visual_cls, losses = self.dual_encoder(audio_feats = batch['audio'], attention_mask = batch['audio_attention_mask'], visual_feats = batch['visual_feats'], visual_pos = batch['visual_pos'], target_list = batch['label'])
        losses = self.solo_module_coarse(audio_cls, visual_cls, batch["img_id"])
        if self.args.fine_matching_weight:
            B = visual_feats.shape[0]
            visual_feats_square = visual_feats.repeat(B,1,1)
            audio_feats_square = audio_feats.repeat_interleave(B, dim=0)
            extended_audio_attention_mask_square = extended_audio_attention_mask.repeat_interleave(B, dim=0)
            audio_cls, visual_cls = self.cross_encoder(audio_feats_square, extended_audio_attention_mask_square, visual_feats_square)
            cross_loss = self.solo_module_fine(audio_cls, visual_cls, batch["img_id"])
            for key, item in cross_loss.items():
                losses[key] = item
        return losses
    
    def train(self):
        flag = True
        step_per_epoch = int(self.train_data_length/self.args.batch_size)
        data_start_time = time.time()

        while flag:
            if self.use_libri_loss:
                libri_loader_iterator = iter(self.libri_train_loader)
            for i, batch in enumerate(self.train_loader):
                if self.use_libri_loss:
                    libri_batch = next(libri_loader_iterator)
                data_end_time = time.time()
                self.dual_encoder.train()
                if self.args.fine_matching_weight != 0:
                    self.cross_encoder.train()
                if self.args.solo_loss:
                    self.solo_module_coarse.train()
                    self.solo_module_fine.train()

                if self.progress['num_updates'] > self.total_num_updates:
                    flag = False
                    self.validate_and_save()
                    self.writer.close()
                    break
                
                cur_lr = np.mean(self.optimizer.get_lr())

                self.writer.add_scalar("lr", cur_lr, self.progress['num_updates'])
                cur_step = self.progress['num_updates'] % step_per_epoch

                cur_batch = {
                        "visual_feats": batch['visual_feats'].to(self.device),
                        "visual_pos": batch['boxes'].to(self.device),
                        "audio": batch['audio'].to(self.device),
                        "audio_attention_mask": batch['audio_attention_mask'].to(self.device),
                        "img_id": batch['img_id'],
                        "label": None
                        }
                if self.args.solo_loss:
                    losses = self.forward_solo(cur_batch)
                else:
                    losses = self.forward(cur_batch)
                if self.use_libri_loss:
                    losses.update(self.dual_encoder(audio_feats = libri_batch['audio'].to(self.device), attention_mask = libri_batch['audio_attention_mask'].to(self.device), target_list = libri_batch['label'], forward_libri=True))

                for key in losses:
                    if key in self.meters:
                        self.meters[key].update(losses[key].mean().cpu().item(), cur_batch['visual_feats'].shape[0])
                        self.writer.add_scalar(key, self.meters[key].val, self.progress['num_updates'])
                
                weighted_loss = self.weight_loss(losses) / self.args.grad_accum

                self.meters['weighted_loss'].update(weighted_loss.item(), cur_batch['visual_feats'].shape[0])
                self.writer.add_scalar('weighted_loss', weighted_loss.item(), self.progress['num_updates'])

                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.trainables, 1.)
                if ((i + 1) % self.args.grad_accum == 0) or (i + 1 == len(self.train_loader)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.meters['data_time'].update(data_end_time - data_start_time)
                self.meters['train_time'].update(time.time() - data_end_time)
                #########
                self.writer.add_scalar("data_time", data_end_time - data_start_time, self.progress['num_updates'])
                self.writer.add_scalar("train_time", time.time() - data_end_time, self.progress['num_updates'])
                # print(self.progress['num_updates'], end=' ', flush=True)
                # logging
                if self.progress['num_updates'] % self.args.n_print_steps == 0:
                    log_out = {}
                    log_out['epoch'] = f"{self.progress['epoch']}/{self.args.n_epochs}"
                    log_out['cur_step/steps_per_epoch'] = f"{cur_step}/{step_per_epoch}"
                    log_out['num_updates'] = self.progress['num_updates']
                    log_out['lr'] = f"{cur_lr:.15f}"
                    for key in self.meters:
                        if self.meters[key].val != 0 or self.meters[key].avg != 0:
                            log_out[key] = f"{self.meters[key].val:.4f} ({self.meters[key].avg:.4f})" if isinstance(self.meters[key].val, float) else f"{self.meters[key].val}"
                    logger.info(log_out)
                    if np.isnan(self.meters['weighted_loss'].avg):
                        logger.info("training diverged...")
                        return
                # validation and save models
                if self.progress['num_updates'] % self.args.n_val_steps == 0:
                    self.validate_and_save(libri=self.use_libri_loss, places=self.args.places)
                self.progress['num_updates'] += 1
                self.progress['epoch'] = int(math.ceil(self.progress['num_updates'] / step_per_epoch))
                data_start_time = time.time()

    def validate_and_save(self, libri=False, places=False):
        self.dual_encoder.eval()
        if self.args.fine_matching_weight != 0:
            self.cross_encoder.eval()
        if self.args.solo_loss:
            self.solo_module_coarse.eval()
            self.solo_module_fine.eval()
        if places:
            r10, r5, r1 = self.validate(self.valid_loader)
            r10_unseen, r5_unseen, r1_unseen = self.validate(self.valid_loader2, unseen=True)
            r10, r5, r1 = (r10+r10_unseen)/2, (r5+r5_unseen)/2, (r1+r1_unseen)/2
        else:
            loss = self.validate_one_to_many()
        
        if libri:
            self.validate_libri()
        # r1 = 0.1 # ignore validation, for debugging
        weighted_loss = self.weight_loss(loss)
        logger.info(f"weighted loss {weighted_loss}")
        if weighted_loss < self.progress['best_acc']:
            self.progress['best_epoch'] = self.progress['epoch']
            self.progress['best_acc'] = weighted_loss
            save_path = os.path.join(self.args.exp_dir,"best_bundle.pth")
            save_dict = {
                "dual_encoder": self.dual_encoder.module.state_dict() if torch.cuda.device_count() > 1 else self.dual_encoder.state_dict(),
                "optimizer":  self.optimizer.state_dict(),
                "indices": self.train_sampler.state_dict(),
                "libri_indices": self.libri_train_sampler.state_dict() if self.libri_train_sampler is not None else None
            }
            if self.args.fine_matching_weight != 0:
                if self.args.solo_loss:
                    save_dict["solo_module_fine"] = self.solo_module_fine.module.state_dict()  if torch.cuda.device_count() > 1 else self.solo_module_fine.state_dict()
                save_dict["cross_encoder"] = self.cross_encoder.module.state_dict() if torch.cuda.device_count() > 1 else self.cross_encoder.state_dict()
                
            if self.args.solo_loss:
                save_dict["solo_module_coarse"] = self.solo_module_coarse.module.state_dict() if torch.cuda.device_count() > 1 else self.solo_module_coarse.state_dict()

            torch.save(save_dict, save_path)
            logger.info(f"save *best* models at {save_path} at global step {self.progress['num_updates']}")
        save_progress(self)
        save_path = os.path.join(self.args.exp_dir,"bundle.pth")
        save_dict = {
            "dual_encoder": self.dual_encoder.module.state_dict() if torch.cuda.device_count() > 1 else self.dual_encoder.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "indices": self.train_sampler.state_dict(),
            "libri_indices": self.libri_train_sampler.state_dict() if self.libri_train_sampler is not None else None
        }
        if self.args.fine_matching_weight != 0:
            if self.args.solo_loss:
                save_dict["solo_module_fine"] = self.solo_module_fine.module.state_dict()  if torch.cuda.device_count() > 1 else self.solo_module_fine.state_dict()
            save_dict["cross_encoder"] = self.cross_encoder.module.state_dict() if torch.cuda.device_count() > 1 else self.cross_encoder.state_dict()
            
        if self.args.solo_loss:
            save_dict["solo_module_coarse"] = self.solo_module_coarse.module.state_dict() if torch.cuda.device_count() > 1 else self.solo_module_coarse.state_dict()

        torch.save(save_dict, save_path)
        logger.info(f"save models, indices, acc and other statistics at {save_path} and {self.args.exp_dir}/progress.pkl at global step {self.progress['num_updates']}")


    def validate_one_to_many(self, hide_progress=True):
        self.dual_encoder.eval()
        if self.args.fine_matching_weight != 0:
            self.cross_encoder.eval()
        if self.args.solo_loss:
            self.solo_module_coarse.eval()
            self.solo_module_fine.eval()

        N_examples = self.valid_loader.dataset.__len__()

        with torch.no_grad():
            # get single modal representations
            audio_feats_total = [] 
            extended_audio_attention_mask_total = []
            audio_cls_total = []
            audio_img_id_total = [] # this is same order as audio_cls_total and audio_feats_total
            img_id_to_img_feats = {}
            img_img_id_list = []
            img_cls_list = [] # this is distinct, order is the same as img_img_id_list
            img_feats_list = [] # this is distinct, order is the same as img_img_id_list
            loss = {}
            for i, batch in enumerate(self.valid_loader):
                self.dual_encoder.eval()
                self.solo_module_coarse.eval()
                if self.args.fine_matching_weight != 0:
                    self.cross_encoder.eval()
                    self.solo_module_fine.eval()
                
                audio_feats, audio_cls, extended_audio_attention_mask, visual_feats, visual_cls = self.dual_encoder(audio_feats = batch['audio'].to(self.device), attention_mask = batch['audio_attention_mask'].to(self.device), visual_feats = batch['visual_feats'].to(self.device), visual_pos = batch['boxes'].to(self.device), test = True)
                losses = self.solo_module_coarse(audio_cls, visual_cls, batch["img_id"])
                if self.args.fine_matching_weight:
                    B = visual_feats.shape[0]
                    visual_feats_square = visual_feats.repeat(B,1,1)
                    audio_feats_square = audio_feats.repeat_interleave(B, dim=0)
                    extended_audio_attention_mask_square = extended_audio_attention_mask.repeat_interleave(B, dim=0)
                    audio_cls, visual_cls = self.cross_encoder(audio_feats_square, extended_audio_attention_mask_square, visual_feats_square)
                    cross_loss = self.solo_module_fine(audio_cls, visual_cls)
                    for key, item in cross_loss.items():
                        losses[key] = item     
                for key in losses.keys():
                    if not loss.get(key):
                        loss[key] = [losses[key].item()]
                    else:
                        loss[key].append(losses[key].item())
            for key in loss:
                loss[key] = np.mean(np.array(loss[key]))
            logger.info(loss)
        return loss
        
    def validate_libri(self):
        with torch.no_grad():
            N = 0
            total_loss = 0
            for batch in self.libri_valid_loader:
                self.dual_encoder.eval()
                n = len(batch['audio'])
                N += n
                losses = self.dual_encoder(audio_feats = batch['audio'].to(self.device), attention_mask = batch['audio_attention_mask'].to(self.device), target_list=batch['label'], forward_libri=True)
                total_loss += losses['libri_w2v2_loss'].mean()*n
        cur_val_loss = (total_loss/N).item()
        self.writer.add_scalar("libri_val_loss", cur_val_loss, self.progress['num_updates'])

        if cur_val_loss < self.progress['best_libri_val_loss']:
            self.progress['best_libri_val_loss'] = cur_val_loss
            logger.info(f"libri validation loss: {cur_val_loss:.3f}*\n")
        else:
            logger.info(f"libri validation loss: {cur_val_loss:.3f}\n")

    def _setup_meters(self):
        meters = {}
        meter_names = ['weighted_loss', "fine_matching_loss", "coarse_matching_loss", 'coarse_sim_loss', 'coarse_std_loss', 'coarse_cov_loss', 'fine_sim_loss', 'fine_std_loss', 'fine_cov_loss', 'caption_w2v2_loss', "libri_w2v2_loss", "caption_hubert_loss", "libri_hubert_loss", "caption_m_acc", "libri_m_acc",'data_time', 'train_time']

        for name in meter_names:
            meters[name] = AverageMeter()
        return meters
    
    def _setup_models(self):
        dual_encoder = fast_vgs.DualEncoder(self.args)
        cross_encoder = fast_vgs.CrossEncoder(self.args)
        print_model_info(dual_encoder)
        print_model_info(cross_encoder)
        if self.args.trained_weights_dir != None:
            bundle = torch.load(os.path.join(self.args.trained_weights_dir, "best_bundle.pth"))
            dual_encoder.carefully_load_state_dict(bundle['dual_encoder'])
            if self.args.fine_matching_weight != 0:
                cross_encoder.carefully_load_state_dict(bundle['cross_encoder'])
                if self.args.solo_loss:
                    self.solo_module_fine.load_state_dict(bundle['solo_module_fine'])
            if self.args.solo_loss:
                self.solo_module_coarse.load_state_dict(bundle['solo_module_coarse'])
            indices = None
            libri_indices = None
            optim_states = None
            # logger.info("loaded parameters and data indices from epoch %d, global step %d" % (self.progress['epoch'], self.progress['num_updates']))
            logger.info(f"Load trained weights from {self.args.trained_weights_dir}")
        elif self.args.validate:
            bundle = torch.load(os.path.join(self.args.exp_dir, "best_bundle.pth"))
            dual_encoder.carefully_load_state_dict(bundle['dual_encoder'])
            if self.args.fine_matching_weight != 0:
                cross_encoder.carefully_load_state_dict(bundle['cross_encoder'])
                if self.args.solo_loss:
                    self.solo_module_fine.load_state_dict(bundle['solo_module_fine'])
            if self.args.solo_loss:
                self.solo_module_coarse.load_state_dict(bundle['solo_module_coarse'])
            indices = None
            libri_indices = None
            optim_states = None
            # logger.info("loaded parameters and data indices from epoch %d, global step %d" % (self.progress['epoch'], self.progress['num_updates']))
            logger.info("Perform Validation")
        elif self.progress['num_updates'] > 1:
            bundle = torch.load(os.path.join(self.args.exp_dir, "bundle.pth"))
            dual_encoder.carefully_load_state_dict(bundle['dual_encoder'])
            if self.args.fine_matching_weight != 0:
                cross_encoder.carefully_load_state_dict(bundle['cross_encoder'])
                if self.args.solo_loss:
                    self.solo_module_fine.load_state_dict(bundle['solo_module_fine'])
            if self.args.solo_loss:
                self.solo_module_coarse.load_state_dict(bundle['solo_module_coarse'])
            indices = bundle['indices']
            libri_indices = bundle['libri_indices']
            optim_states = bundle['optimizer']
            logger.info("loaded parameters and data indices from epoch %d, global step %d" % (self.progress['epoch'], self.progress['num_updates']))
        else:
            indices = None
            libri_indices = None
            optim_states = None

        if self.args.fb_w2v2_weights_fn and self.progress['num_updates'] <= 1 and not self.args.validate and self.args.trained_weights_dir == None:
            dual_encoder.conv1_trm1_trm3.carefully_load_state_dict(torch.load(self.args.fb_w2v2_weights_fn)['model'])

        if self.args.feature_grad_mult <= 0.:
            for name, p in dual_encoder.named_parameters():
                if "feature_extractor" in name:
                    p.requires_grad = False
        trainables1 = [p for p in dual_encoder.parameters() if p.requires_grad]
        trainables2 = [p for p in cross_encoder.parameters() if p.requires_grad]
        trainables = trainables1 + trainables2
        if self.args.solo_loss:
            trainables3 = [p for p in self.solo_module_coarse.parameters() if p.requires_grad]
            trainables4 = [p for p in self.solo_module_fine.parameters() if p.requires_grad]
            trainables += trainables3 + trainables4

        dual_encoder.to(self.device)
        if self.args.fine_matching_weight != 0:
            cross_encoder.to(self.device)
        else:
            cross_encoder = None
        return dual_encoder, cross_encoder, trainables, indices, libri_indices, optim_states
    def _setup_dataloader(self):
        if self.args.places:
            # raise NotImplementedError
            train_dataset = places_dataset.ImageCaptionDataset(self.args, split='train')
            val_seen_dataset = places_dataset.ImageCaptionDataset(self.args, split='val_seen')
            val_unseen_dataset = places_dataset.ImageCaptionDataset(self.args, split='val_unseen')
            train_sampler = StatefulSampler(len(train_dataset))
            if self.progress['num_updates'] > 1 and self.indices is not None:
                train_sampler.load_state_dict(self.indices)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True, sampler = train_sampler, collate_fn = train_dataset.collate, drop_last=True)
            valid_loader = torch.utils.data.DataLoader(val_seen_dataset, batch_size=self.args.val_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, collate_fn = val_seen_dataset.collate)
            valid_loader2 = torch.utils.data.DataLoader(val_unseen_dataset, batch_size=self.args.val_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, collate_fn = val_unseen_dataset.collate)
        elif self.args.flickr8k:
            train_dataset = flickr8k_dataset.ImageCaptionDataset(self.args, split='train')
            val_dataset = flickr8k_dataset.ImageCaptionDataset(self.args, split='val')
            train_sampler = StatefulSampler(len(train_dataset))
            if self.progress['num_updates'] > 1 and self.indices is not None:
                train_sampler.load_state_dict(self.indices)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True, sampler = train_sampler, collate_fn = train_dataset.collate, drop_last=True)
            valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.val_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, collate_fn = val_dataset.collate)
            valid_loader2 = None
        else:
        # SpokenCOCO
            train_dataset = spokencoco_dataset.ImageCaptionDataset(self.args, split='train')
            val_dataset = spokencoco_dataset.ImageCaptionDataset(self.args, split='val')
            train_sampler = StatefulSampler(len(train_dataset))
            if self.progress['num_updates'] > 1 and self.indices is not None:
                train_sampler.load_state_dict(self.indices)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True, sampler = train_sampler, collate_fn = train_dataset.collate, drop_last=True)
            valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.val_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, collate_fn = val_dataset.collate)
            valid_loader2 = None

        if self.use_libri_loss:
            # librispeech dataloaders
            # train
            step_per_epoch = int(np.floor(len(train_dataset)/self.args.batch_size))
            # libri_train_dataset = libri_dataset_mm.LibriDataset(self.args, split="train")
            libri_train_dataset = libri_dataset.LibriDataset(self.args, split="train")
            libri_train_bzs = libri_train_dataset.calculate_batch_size(step_per_epoch)
            libri_train_bzs = min(libri_train_bzs, 64)
            logger.info(f"librispeech train batch size: {libri_train_bzs}")
            libri_train_sampler = StatefulSampler(len(libri_train_dataset))
            if self.progress['num_updates'] > 1 and self.libri_indices is not None:
                libri_train_sampler.load_state_dict(self.libri_indices)
            libri_train_loader = torch.utils.data.DataLoader(libri_train_dataset, batch_size=libri_train_bzs, num_workers=self.args.num_workers, pin_memory=True, sampler = libri_train_sampler, collate_fn = libri_train_dataset.collate, drop_last=True)
            
            # val
            # libri_val_dataset = libri_dataset_mm.LibriDataset(self.args, split="val")
            libri_val_dataset = libri_dataset.LibriDataset(self.args, split="val")
            logger.info(f"librispeech val batch size: {self.args.libri_val_bzs}")
            libri_valid_loader = torch.utils.data.DataLoader(libri_val_dataset, batch_size=self.args.libri_val_bzs, num_workers=self.args.num_workers, pin_memory=True, collate_fn = libri_val_dataset.collate, drop_last=True)
        else:
            libri_train_loader = None
            libri_valid_loader = None
            libri_train_sampler = None

        return train_loader, valid_loader, valid_loader2, train_sampler, libri_train_loader, libri_valid_loader, libri_train_sampler, len(train_dataset)
    
    def _setup_optimizer(self):
        optimizer = BertAdam(self.trainables, lr=self.args.lr, warmup=self.args.warmup_fraction, t_total=self.total_num_updates)

        if self.progress['num_updates'] > 1:
            optimizer.load_state_dict(self.optim_states)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        optimizer.zero_grad()
        return optimizer
    
    def _setup_scheduler(self):
        pass

    def weight_loss(self, losses):
        if self.args.solo_loss:
            weighted_loss = losses['coarse_sim_loss'] * self.args.sim_coeff_coarse + losses['coarse_std_loss'] * self.args.std_coeff_coarse + losses['coarse_cov_loss'] * self.args.cov_coeff_coarse
            if self.args.fine_matching_weight != 0:
                weighted_loss += losses['fine_sim_loss'] * self.args.sim_coeff_fine + losses['fine_std_loss'] * self.args.std_coeff_fine + losses['fine_cov_loss'] * self.args.cov_coeff_fine
            if 'coarse_matching_loss' in losses:
                weighted_loss += losses['coarse_matching_loss'] * self.args.coarse_matching_weight
            if 'fine_matching_loss' in losses:
                weighted_loss += losses['fine_matching_loss'] * self.args.fine_matching_weight
        else:
            weighted_loss = losses['coarse_matching_loss'] * self.args.coarse_matching_weight
            if 'fine_matching_loss' in losses:
                weighted_loss += losses['fine_matching_loss'] * self.args.fine_matching_weight
            if 'caption_w2v2_loss' in losses:
                weighted_loss += losses['caption_w2v2_loss'].mean() * self.args.caption_w2v2_weight
            if 'libri_w2v2_loss' in losses:
                weighted_loss += losses['libri_w2v2_loss'].mean() * self.args.libri_w2v2_weight
            if 'caption_hubert_loss' in losses:
                weighted_loss += losses['caption_hubert_loss'].mean() * self.args.caption_hubert_weight
            if 'libri_hubert_loss' in losses:
                weighted_loss += losses['libri_hubert_loss'].mean() * self.args.libri_hubert_weight
            
        return weighted_loss
    
    def seed_everything(self, seed=1):
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

