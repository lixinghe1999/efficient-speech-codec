import torch, torchaudio
import transformers
import numpy as np
import argparse, yaml, glob
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset, DataLoader, default_collate, ConcatDataset

from esc.modules import ComplexSTFTLoss, MelSpectrogramLoss


def quantization_dropout(dropout_rate: float, max_streams: int):
    """
    Args:
        dropout_rate: probability that applies quantization dropout 
        max_streams: maximum number of streams codec can take
        returns: sampled number of streams for current batch
    """
    assert dropout_rate >=0 and dropout_rate <=1, "dropout_rate must be within [0, 1]"
    # Do Random Sample N w prob dropout_rate
    do_sample = np.random.choice([0, 1], p=[1-dropout_rate, dropout_rate])
    if do_sample: 
        streams = np.random.randint(1, max_streams+1)
    else:
        streams = max_streams
    return streams

class EvalSet(Dataset):
    def __init__(self, eval_folder_path, duration=10,sample_rate=16000, hop_len=5) -> None:
        super().__init__()
        self.testset_files = glob.glob(f"{eval_folder_path}/*.wav")
        if not self.testset_files:
            self.testset_files = glob.glob(f"{eval_folder_path}/*/*.wav")
        self.testset_files = self.testset_files[:180000]
        self.duration = duration
        self.sample_rate = sample_rate
        self.hop_len = int(hop_len * sample_rate // 1000)  # hop_len in samples
        self.win_len = 4 * self.hop_len  # win_len in samples, 4k-1
    def __len__(self):
        return len(self.testset_files)

    def __getitem__(self, i):
        file_path = self.testset_files[i]
        # Load audio file by randomly crop to self.duration seconds
        metadata = torchaudio.info(file_path)
        max_frames = metadata.num_frames
        if max_frames <= self.duration * self.sample_rate:
            x, sr = torchaudio.load(file_path)
        else:
            random_offset = np.random.randint(0, max_frames - self.duration * self.sample_rate)
            # random_offset = 0
            x, sr = torchaudio.load(file_path, frame_offset=random_offset, num_frames=self.duration * self.sample_rate)
        if len(x.shape) == 2:
            x = x[0]
        x = torch.nn.functional.pad(x, (0, self.duration * self.sample_rate - x.shape[-1]), mode='constant', value=0)
        x = torch.nn.functional.pad(x, (0, self.win_len - x.shape[-1] % self.win_len - self.hop_len), mode='constant', value=0)
        return x
    
def make_dataloader(data_path, batch_size, shuffle, num_workers=0, sample_rate=16000, hop_len=50):
    if isinstance(data_path, str):
        ds = EvalSet(data_path, duration=10, sample_rate=sample_rate, hop_len=hop_len)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        collate_fn=default_collate, num_workers=num_workers)
    elif isinstance(data_path, list):
        dss = [EvalSet(dp, duration=10, sample_rate=sample_rate, hop_len=hop_len) for dp in data_path]
        ds = ConcatDataset(dss)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, 
                        collate_fn=default_collate, num_workers=num_workers)
    else:
        raise ValueError("data_path must be a string or a list of strings")
    return dl

def make_optimizer(params, lr):
    return torch.optim.AdamW(params, lr)

GAMMAR = 0.999996
def make_scheduler(optimizer, scheduler_type, total_steps=250000, warmup_steps=0):
    if scheduler_type == "constant":
        scheduler = transformers.get_constant_schedule(optimizer)
    elif scheduler_type == "constant_warmup":
        scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps) 
    elif scheduler_type == "cosine_warmup":
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    elif scheduler_type == "exponential_decay":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMAR)
    else:
        raise ValueError("\{scheduler_type\} must be in ('constant', 'constant_warmup', 'cosine_warmup', 'exponential_decay')")
    return scheduler

def make_losses(name="mel_loss"):
    if name == "mel_loss":
        return MelSpectrogramLoss()
    elif name == "stft_loss":
        return ComplexSTFTLoss(power_law=True)
    else:
        raise ValueError("Supported losses are (mel_loss, stft_loss)")
    
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def namespace2dict(config):
    return vars(config)

def read_yaml(pth):
    with open(pth, 'r') as f:
        config = yaml.safe_load(f)
    return config

def download_data_hf(repo_id="../dnscustom", 
                     filename="testset.tar.gz",
                     local_dir="./data"):

    file_path = hf_hub_download(repo_id=repo_id, 
                                filename=filename, 
                                repo_type="dataset",
                                local_dir=local_dir)
    print(f"File has been downloaded and is located at {file_path}")
    return file_path
