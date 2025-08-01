from .metrics import PESQ, MelSpectrogramDistance, SISDR, LSD
from .utils import read_yaml, EvalSet
from baselines.descript.dac.model.dac import DAC

from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm
import numpy as np

import argparse, torch, json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_folder_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--model_path", type=str, required=True, help="folder contains model configuration and checkpoint")
    parser.add_argument("--save_path", type=str, default=None, help="folder to save test statistics")

    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()

@torch.no_grad()
def eval_epoch(model, eval_loader: DataLoader, 
               metric_funcs: dict, device: str,
               verbose: bool = True):
    model.eval()

    # Initialize performance tracking
    perf = {k: [] for k in metric_funcs.keys()}
    
    for _, x in tqdm(enumerate(eval_loader), total=len(eval_loader), desc="Evaluating DAC Model"):
        x = x.to(device)
        # DAC model has slightly different interface
        outputs = model(x.unsqueeze(1))  # Add channel dimension
        recon_x = outputs["audio"]
        
        for k, func in metric_funcs.items():    
            perf[k].extend(func(x, recon_x.squeeze(1)).tolist())  # Remove channel dimension for metrics

    # Calculate mean metrics
    mean_perf = {k: round(np.mean(v), 4) for k, v in perf.items()}

    if verbose:
        print("Test Metrics: ", end="")
        print(" | ".join(f"{k}: {v:.4f}" for k, v in mean_perf.items()))

    model.train()
    return mean_perf

def run(args):
    # Metrics
    metric_funcs = {
        "PESQ": PESQ(), 
        "LSD": LSD().to(args.device),
        "MelDistance": MelSpectrogramDistance().to(args.device), 
        "SISDR": SISDR().to(args.device)
    }

    # Model
    cfg = read_yaml(f"{args.model_path}/config.yaml")

    # Data
    eval_set = EvalSet(args.eval_folder_path, duration=10, sample_rate=cfg['model']['sample_rate'])
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, collate_fn=default_collate)

    model = DAC(**cfg['model'])
    model.load_state_dict(
        torch.load(f"{args.model_path}/weights.pth", map_location="cpu")["state_dict"],
    )
    model = model.to(args.device)

    performances = eval_epoch(
        model, eval_loader, metric_funcs, args.device, verbose=True
    )
    
    save_path = args.model_path if args.save_path is None else args.save_path
    json.dump(performances, open(f"{save_path}/perf_stats.json", "w"), indent=2)
    print(f"Test statistics saved into {save_path}/perf_stats.json")

if __name__ == "__main__":
    args = parse_args()
    run(args)