import torchaudio, torch
from esc import ESC
import yaml
import soundfile as sf
import os
import time
import argparse
from baselines.descript.dac.model.dac import DAC
from calflops import calculate_flops_hf, calculate_flops


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for ESC model")
    parser.add_argument("--mode", type=str, choices=["9kbps", "27kbps", "dac"])
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    parser.add_argument("--num_inference_steps", type=int, default=100, help="Number of inference steps to run")
    args = parser.parse_args()

    device = args.device
    duration = 10  # seconds
    num_inference_steps = args.num_inference_steps
    if args.mode == "9kbps":
        config_path = "checkpoints/esc_base_adv/esc9kbps_base_adversarial/config.yaml"
        ckpt_path = "checkpoints/esc_base_adv/esc9kbps_base_adversarial/model.pth"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model = ESC(**config["model"])
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model_state_dict"],)
    elif args.mode == "27kbps":
        config_path = "configs/27kbps_esc_base_adv.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model = ESC(**config["model"])
    else:
        config_path = "checkpoints/dac_tiny_adv/dac9kbps_tiny_adversarial/config.yaml"
        ckpt_path = "checkpoints/dac_tiny_adv/dac9kbps_tiny_adversarial/weights.pth"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        model = DAC(**config["model"])
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"],)

    x, sr = torchaudio.load("test_dataset/ori_44kHz_mono_wav/101167817.wav")

    if args.mode in ["9kbps", "27kbps"]:
        sr = config["model"]["sr"]
        win_len = config["model"]["win_len"] * sr // 1000; hop_len = config["model"]["hop_len"] * sr // 1000
        x = torchaudio.functional.resample(x, sr, sr)  # resample to 16000Hz
        # truncate to 10 seconds
        x = x[:, :duration*sr]  # mono channel, first 10 seconds
        # pad to 80(4k-1)
        x = torch.nn.functional.pad(x, (0, win_len - x.shape[-1] % win_len - hop_len), mode='constant', value=0)
        # Enc. (@ num_streams*1.5 kbps)
    else:
        x = x[None, :, :duration*sr]  

    model.to(device); model.eval()
    x = x.to(device)

    if args.mode in ["9kbps", "27kbps"]:
        folder = os.path.join("output", f"esc_{args.mode}")
        os.makedirs(folder, exist_ok=True)
        for _num_streams in range(1,7):
            with torch.no_grad():
                t_start = time.time()
                for i in range(num_inference_steps):
                    codes, f_shape = model.encode(x, num_streams=_num_streams)
                    recon_x = model.decode(codes, f_shape)
                t_end = time.time()
                print(f"Encoding and decoding with {_num_streams}, RTF is {(t_end - t_start) / num_inference_steps / duration:.4f}.")
            # save the reconstructed audio
            sf.write(os.path.join(folder, f'recon_{_num_streams}.wav'), recon_x.squeeze().cpu().numpy(), sr)
        sf.write(os.path.join(folder, "input.wav"), x.squeeze().cpu().numpy(), sr)
        flops, macs, params = calculate_flops(model=model, input_shape=tuple(x.shape), print_detailed=False)
        print(f"Model FLOPs: {flops}, MACs: {macs}, Params: {params}")
    else:
        t_start = time.time()
        with torch.no_grad():
            for _ in range(num_inference_steps):
                out = model(x)["audio"]
        t_end = time.time()
        print(f"DAC RTF: {(t_end - t_start) / num_inference_steps / duration:.4f}.")
        flops, macs, params = calculate_flops(model=model, input_shape=tuple(x.shape), print_detailed=False)
        print(f"Model FLOPs: {flops}, MACs: {macs}, Params: {params}")
