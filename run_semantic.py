import time
import os
import torch
import math
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datasets.zerospeech_dataset import ZerospeechDataset
from datasets.sampler import StatefulSampler
from steps.utils import *
from steps.trainer_utils import *
from steps.bert_adam import BertAdam
from logging import getLogger
import soundfile as sf
from models import fast_vgs, w2v2_model
from datasets import flickr8k_dataset, libri_dataset
import argparse

logger = getLogger(__name__)

class Task_base():
    TASKS_NAME = ["lexical", "phonetic",  "semantic",  "syntactic"]
    def __init__(self,my_model,output_result_dir,inference_bsz=64,sample_rate=16_000,run_dev=False,run_test=False,**kwarg):
        self.run_dev = run_dev
        self.run_test = run_test
        self.sample_rate = sample_rate
        self.inference_bsz = inference_bsz
        self.my_model = my_model
        self.output_result_dir = output_result_dir

        if os.path.exists(self.output_result_dir):
            print("output_result_dir({}) exists!".format(self.output_result_dir))
            exit(1)
        else:
            os.makedirs(self.output_result_dir,exist_ok=True)

        for _tn in self.TASKS_NAME:
            os.makedirs(os.path.join(self.output_result_dir,_tn),exist_ok=True)

        # create meta yaml

        shutil.copy("./meta.yaml",os.path.join(self.output_result_dir,"meta.yaml"))


class zerospeech:
    DATA_SOURCE_NAME = ["librispeech","synthetic"]
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
        parser.add_argument("--same_projector", action="store_true", default=False, help="Same projector for audio and image")
        parser.add_argument("--sim_coeff", type=float, default=25.0, help="Invariance regularization loss coefficient for VICReg")
        parser.add_argument("--std_coeff", type=float, default=25.0, help="Variance regularization loss coefficient for VICReg")
        parser.add_argument("--cov_coeff", type=float, default=1.0, help="Covariance regularization loss coefficient for VICReg")
        parser.add_argument("--lambd", type=float, default=0.0051, help="Weight on off-diagonal terms for barlow twins")
        parser.add_argument("--output_result_dir", type=str)
        parser.add_argument("--task_name", type=str)
        
    def __init__(self, args):
        self.start_time = time.time()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"number of devices: {torch.cuda.device_count()}")
        self.writer = SummaryWriter(self.args.exp_dir)
        self.seed_everything(seed=self.args.seed)
        self.progress, self.total_progress = setup_progress(self)
        
        if self.args.solo_loss == 'VICReg':
            self.solo_module_coarse = fast_vgs.VICReg(self.args, 'coarse').to(self.device)
            self.solo_module_fine = fast_vgs.VICReg(self.args, 'fine').to(self.device)
        elif self.args.solo_loss == 'BarlowTwins':
            self.solo_module_coarse = fast_vgs.BarlowTwins(self.args).to(self.device)
            self.solo_module_fine = fast_vgs.BarlowTwins(self.args).to(self.device)
           
        self.dual_encoder, self.cross_encoder, self.trainables, self.indices, self.libri_indices, self.optim_states = self._setup_models()
        if torch.cuda.device_count() > 1:
            self.dual_encoder = nn.DataParallel(self.dual_encoder)
            self.cross_encoder = nn.DataParallel(self.cross_encoder)

        self.task_input_dir = args.task_input_dir
        self.task_name = args.task_name
        self.task_root_dir = os.path.join(args.output_result_dir, args.task_name)
        assert os.path.exists(self.task_root_dir)
        self.audio_wav_paths = {}
        
        os.makedirs(os.path.join(self.task_root_dir,"dev"),exist_ok=True)
        os.makedirs(os.path.join(self.task_root_dir,"dev","librispeech"),exist_ok=True)
        os.makedirs(os.path.join(self.task_root_dir,"dev","synthetic"),exist_ok=True)
        
    def run(self):
        self.dual_encoder.eval()
        print(f"Inferencing zerospeech {self.task_name} dev")
        for data_source in self.DATA_SOURCE_NAME:
            dataset = ZerospeechDataset(self.args, 'dev', data_source)
            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=self.args.val_batch_size, 
                shuffle=False, 
                num_workers=self.args.num_workers, 
                pin_memory=True, 
                collate_fn = dataset.collate
            )
            print("Datasource: {}, total:{}".format(
                data_source,
                dataset.__len__()
            ))

            for batch in tqdm(dataloader):
                with torch.no_grad():
                    cur_batch = {
                        "audio": batch['audio'].to(self.device),
                        "audio_length": batch['audio_length'],
                        "audio_attention_mask": batch['audio_attention_mask'].to(self.device),
                        "path": batch['path']
                    }
                    embeddings, _, _, _ = self.dual_encoder.forward_audio(cur_batch["audio"], cur_batch["audio_attention_mask"], True, target_list=None)
                    embeddings = embeddings.cpu().float().numpy()

                for i, embed in enumerate(embeddings):
                    txt_path = os.path.join(self.task_root_dir, "dev", data_source, os.path.basename(cur_batch['path'][i]).replace(".wav",".txt"))
                    np.savetxt(txt_path, embed)
        print(f"Done inferencing zerospeech {self.task_name} dev")
    
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
    
    def seed_everything(self, seed=1):
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True



class Task_semantic(Task_base):
    DATA_SOURCE_NAME = ["librispeech","synthetic"]
    def __init__(
        self,
        model_cls_name,
        model_ckpt,
        task_input_dir,
        task_name,
        **kwarg):
        super().__init__(**kwarg)
        self.model_cls_name = model_cls_name
        self.model_ckpt = model_ckpt
        self.task_input_dir = task_input_dir
        self.task_name = task_name
        

        self.task_root_dir = os.path.join(self.output_result_dir,"semantic")

        assert os.path.exists(self.task_root_dir)



        self.audio_wav_paths = {}
        if self.run_dev:
            dev_libri_path_txt = os.path.join(self.task_input_dir,"dev_librispeech.txt")
            dev_synthetic_path_txt = os.path.join(self.task_input_dir,"dev_synthetic.txt")
            os.makedirs(os.path.join(self.task_root_dir,"dev"),exist_ok=True)
            os.makedirs(os.path.join(self.task_root_dir,"dev","librispeech"),exist_ok=True)
            os.makedirs(os.path.join(self.task_root_dir,"dev","synthetic"),exist_ok=True)
            
            with open(dev_libri_path_txt,"r") as f:
                _data = [ os.path.join(self.task_input_dir,"dev","librispeech",x.strip())    for x in f.readlines() if x.strip().endswith(".wav")]
                self.audio_wav_paths["dev_librispeech"] = _data

            with open(dev_synthetic_path_txt,"r") as f:
                _data = [os.path.join(self.task_input_dir,"dev","synthetic", x.strip()) for x in f.readlines() if x.strip().endswith(".wav")]
                self.audio_wav_paths["dev_synthetic"] = _data
        

    def run(self):
        self.my_model.eval()
        self.my_model.cuda()
        if self.run_dev:
            print(f"Inferencing zerospeech {self.task_name} dev")
            for data_source in self.DATA_SOURCE_NAME:
                print("Datasource: {}, total:{}, bsz:{} ".format(
                    data_source,
                    len(self.audio_wav_paths["{}_{}".format("dev",data_source)]),
                    self.inference_bsz,
                    ceil(len(self.audio_wav_paths["{}_{}".format("dev",data_source)]) / self.inference_bsz)
                ))
                # _dataset = AudioDataset(
                #     wav_paths=self.audio_wav_paths["{}_{}".format("dev",data_source)],
                #     sr=self.sample_rate
                # )
                # dev_dataloader = DataLoader(
                #     dataset=_dataset,
                #     batch_size=self.inference_bsz,
                #     shuffle=False,
                #     num_workers=8,
                # )

                for i in tqdm.tqdm(range(0,len(self.audio_wav_paths["{}_{}".format("dev",data_source)])+self.inference_bsz,self.inference_bsz)):
                # for _data,_wavpaths in tqdm.tqdm(dev_dataloader):
                    _wavpaths = self.audio_wav_paths["{}_{}".format("dev",data_source)][i:i+self.inference_bsz+self.inference_bsz]
                    if len(_wavpaths) == 0 : continue
                    _data = []
                    for _w in _wavpaths:
                        _data.append(
                            torch.FloatTensor(librosa.load(_w, sr=self.sample_rate)[0]).cuda()
                        )
                    # _data = [x.cuda() for x in _data]
                    with torch.no_grad():
                        embeddings = self.my_model.feature_extractor_zerospeech(wav=_data)
                        embeddings = embeddings.cpu().float().numpy()

                    # print(embeddings.shape)
                    for _embs, _wavpath in zip(embeddings,_wavpaths):
                        # print("embs",_embs.shape)
                        txt_path = os.path.join(self.task_root_dir,"dev",data_source,os.path.basename(_wavpath).replace(".wav",".txt"))
                        np.savetxt(txt_path,_embs)

            print(f"Done inferencing zerospeech {self.task_name} dev")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--resume", action="store_true", dest="resume", help="load from exp_dir if True")
    parser.add_argument("--validate", action="store_true", default=False, help="temp, if call trainer_variants rather than trainer")
    ZerospeechDataset.add_args(parser)
    zerospeech.add_args(parser)
    w2v2_model.Wav2Vec2Model_cls.add_args(parser)
    fast_vgs.DualEncoder.add_args(parser)
    args = parser.parse_args()

    task = zerospeech(args)
    task.run()
