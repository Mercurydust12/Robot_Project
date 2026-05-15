# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from distributed import init_distributed
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import yaml
import argparse
import os
import numpy as np

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

import misc
import distributed as dist
from models import CDiT_models
from datasets import EvalDataset
from PIL import Image


def resolve_checkpoint_path(config, args):
    if args.checkpoint_path is not None:
        checkpoint_path = args.checkpoint_path
    else:
        checkpoint_path = f'{config["results_dir"]}/{config["run_name"]}/checkpoints/{args.ckp}.pth.tar'
    checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    print(f"Loading checkpoint from: {checkpoint_path}")
    return checkpoint_path


def resolve_onnx_model_path(args):
    if args.onnx_model_path is None:
        raise ValueError("--onnx-model-path is required when --denoiser-backend onnx.")
    onnx_model_path = os.path.abspath(os.path.expanduser(args.onnx_model_path))
    print(f"Loading ONNX denoiser from: {onnx_model_path}")
    return onnx_model_path


class ONNXCDiTWrapper:
    EXPECTED_INPUTS = {
        "x": [None, 4, 28, 28],
        "t": [None],
        "y": [None, 3],
        "x_cond": [None, 4, 4, 28, 28],
        "rel_t": [None],
    }
    EXPECTED_OUTPUTS = {"output": [None, 8, 28, 28]}
    EXPECTED_INPUT_ORDER = ["x", "t", "y", "x_cond", "rel_t"]
    NUMPY_DTYPES = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
    }
    TORCH_DTYPES = {
        np.float32: torch.float32,
        np.float16: torch.float16,
        np.int64: torch.long,
        np.int32: torch.int32,
    }

    def __init__(self, model_path, provider):
        import onnxruntime as ort

        self.model_path = model_path
        self.provider = provider
        self.logged_first_call = False

        print(f"ONNX Runtime version: {ort.__version__}")
        print(f"ONNX model path: {model_path}")
        available_providers = ort.get_available_providers()
        print(f"ONNX Runtime providers available: {available_providers}")
        if provider not in available_providers:
            raise ValueError(
                f"Requested ONNX Runtime provider '{provider}' is not available. "
                f"Available providers: {available_providers}"
            )

        self.session = ort.InferenceSession(model_path, providers=[provider])
        self.inputs_meta = self.session.get_inputs()
        self.outputs_meta = self.session.get_outputs()
        self.input_shapes = {inp.name: list(inp.shape) for inp in self.inputs_meta}
        self.input_types = {inp.name: inp.type for inp in self.inputs_meta}
        self.input_names = [inp.name for inp in self.inputs_meta]
        self.output_shapes = {out.name: list(out.shape) for out in self.outputs_meta}
        self.output_names = [out.name for out in self.outputs_meta]
        self.dynamic_batch, self.fixed_batch_size = self._validate_model_metadata()
        print(f"ONNX Runtime provider selected: {provider}")
        print(f"ONNX Runtime providers used: {self.session.get_providers()}")
        print(f"Detected ONNX input shapes: {self.input_shapes}")
        print(f"Detected ONNX input types: {self.input_types}")
        print(f"Detected ONNX output shapes: {self.output_shapes}")
        print(f"ONNX batch dimension is dynamic: {self.dynamic_batch}")
        if self.dynamic_batch:
            print("Allowed eval batch size: dynamic positive batch size")
        else:
            print(f"Allowed eval batch size: {self.fixed_batch_size}")

    def _is_dynamic_dim(self, dim):
        return dim is None or isinstance(dim, str)

    def _validate_shape(self, name, shape, expected):
        if len(shape) != len(expected):
            raise ValueError(f"ONNX '{name}' rank mismatch: expected {expected}, got {shape}.")
        for axis, (dim, expected_dim) in enumerate(zip(shape, expected)):
            if axis == 0:
                continue
            if self._is_dynamic_dim(dim):
                raise ValueError(
                    f"ONNX '{name}' has dynamic non-batch dimension at axis {axis}: {shape}. "
                    "Only batch dimension may be dynamic."
                )
            if int(dim) != int(expected_dim):
                raise ValueError(
                    f"ONNX '{name}' shape mismatch at axis {axis}: expected {expected_dim}, got {dim}. "
                    f"Full shape: {shape}."
                )

    def _validate_model_metadata(self):
        if self.input_names != self.EXPECTED_INPUT_ORDER:
            raise ValueError(f"ONNX input order mismatch: expected {self.EXPECTED_INPUT_ORDER}, got {self.input_names}.")
        if len(self.output_names) != 1 or self.output_names[0] != "output":
            raise ValueError(f"ONNX output mismatch: expected ['output'], got {self.output_names}.")

        batch_dims = []
        for name, expected_shape in self.EXPECTED_INPUTS.items():
            shape = self.input_shapes[name]
            self._validate_shape(name, shape, expected_shape)
            input_type = self.input_types[name]
            if input_type not in self.NUMPY_DTYPES:
                raise ValueError(f"Unsupported ONNX input dtype for '{name}': {input_type}.")
            batch_dims.append(shape[0])
        for name, expected_shape in self.EXPECTED_OUTPUTS.items():
            self._validate_shape(name, self.output_shapes[name], expected_shape)
            batch_dims.append(self.output_shapes[name][0])

        dynamic_flags = [self._is_dynamic_dim(dim) for dim in batch_dims]
        if any(dynamic_flags):
            if not all(dynamic_flags):
                raise ValueError(f"ONNX model mixes dynamic and fixed batch dimensions: {batch_dims}.")
            return True, None

        fixed_sizes = {int(dim) for dim in batch_dims}
        if len(fixed_sizes) != 1:
            raise ValueError(f"ONNX model has inconsistent fixed batch dimensions: {batch_dims}.")
        return False, fixed_sizes.pop()

    def _numpy_input(self, tensor, name):
        numpy_dtype = self.NUMPY_DTYPES[self.input_types[name]]
        torch_dtype = self.TORCH_DTYPES[numpy_dtype]
        tensor = tensor.detach().to(device="cpu", dtype=torch_dtype)
        array = tensor.numpy()
        return array.astype(numpy_dtype, copy=False)

    def _check_runtime_shapes(self, tensors):
        batch_size = None
        for name, tensor in tensors.items():
            shape = self.input_shapes[name]
            actual_shape = tuple(tensor.shape)
            if len(actual_shape) != len(shape):
                raise ValueError(f"Runtime input '{name}' rank mismatch: ONNX shape {shape}, runtime shape {actual_shape}.")
            if batch_size is None:
                batch_size = int(actual_shape[0])
            elif int(actual_shape[0]) != batch_size:
                raise ValueError(f"Runtime input '{name}' batch {actual_shape[0]} does not match batch {batch_size}.")
            for axis, dim in enumerate(shape):
                if axis == 0:
                    continue
                if int(actual_shape[axis]) != int(dim):
                    raise ValueError(
                        f"Runtime input '{name}' shape mismatch at axis {axis}: "
                        f"ONNX expects {dim}, got {actual_shape[axis]}. Full runtime shape: {actual_shape}."
                    )
        if not self.dynamic_batch and batch_size != self.fixed_batch_size:
            raise ValueError(
                f"ONNX denoiser is exported with fixed batch size {self.fixed_batch_size}, "
                f"but runtime batch size is {batch_size}. Re-run with --batch_size {self.fixed_batch_size} "
                "or use a dynamic-batch ONNX model."
            )
        return batch_size

    def forward(self, x, t, y, x_cond, rel_t):
        with torch.no_grad():
            tensors = {"x": x, "t": t, "y": y, "x_cond": x_cond, "rel_t": rel_t}
            self._check_runtime_shapes(tensors)
            feeds = {
                "x": self._numpy_input(x, "x"),
                "t": self._numpy_input(t, "t"),
                "y": self._numpy_input(y, "y"),
                "x_cond": self._numpy_input(x_cond, "x_cond"),
                "rel_t": self._numpy_input(rel_t, "rel_t"),
            }
            feeds = {name: feeds[name] for name in self.EXPECTED_INPUT_ORDER}
            outputs = self.session.run(None, feeds)
            if len(outputs) != 1:
                raise ValueError(f"Expected one ONNX denoiser output, got {len(outputs)}.")

            output = torch.from_numpy(outputs[0]).to(device=x.device, dtype=x.dtype)
            if not self.logged_first_call:
                input_shapes = {name: tuple(value.shape) for name, value in feeds.items()}
                print(f"First ONNX denoiser input shapes: {input_shapes}")
                print(f"First ONNX denoiser output shape: {tuple(output.shape)}")
                self.logged_first_call = True
            return output


def save_image(output_file, img, unnormalize_img):
    img = img.detach().cpu()
    if unnormalize_img:
        img = misc.unnormalize(img)
        
    img = img * 255
    img = img.byte()
    image = Image.fromarray(img.permute(1, 2, 0).numpy(), mode='RGB')

    image.save(output_file)
    
    
def get_dataset_eval(config, dataset_name, eval_type, predefined_index=True):
    data_config = config["eval_datasets"][dataset_name]    
    if predefined_index:
        predefined_index = f"data_splits/{dataset_name}/test/{eval_type}.pkl"
    else:
        predefined_index=None

    
    dataset = EvalDataset(
                data_folder=data_config["data_folder"],
                data_split_folder=data_config["test"],
                dataset_name=dataset_name,
                image_size=config["image_size"],
                min_dist_cat=config["eval_distance"]["eval_min_dist_cat"],
                max_dist_cat=config["eval_distance"]["eval_max_dist_cat"],
                len_traj_pred=config["eval_len_traj_pred"],
                traj_stride=config["traj_stride"], 
                context_size=config["eval_context_size"],
                normalize=config["normalize"],
                transform=misc.transform,
                goals_per_obs=4,
                predefined_index=predefined_index,
                traj_names='traj_names.txt'
            )
    
    return dataset

@torch.no_grad()
def model_forward_wrapper(all_models, curr_obs, curr_delta, num_timesteps, latent_size, device, num_cond, num_goals=1, rel_t=None, progress=False):
    model, diffusion, vae = all_models
    x = curr_obs.to(device)
    y = curr_delta.to(device)

    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        B, T = x.shape[:2]

        if rel_t is None:
            rel_t = (torch.ones(B)* (1. / 128.)).to(device)
            rel_t *= num_timesteps

        x = x.flatten(0,1)
        x = vae.encode(x).latent_dist.sample().mul_(0.18215).unflatten(0, (B, T))
        x_cond = x[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, x.shape[2], x.shape[3], x.shape[4]).flatten(0, 1)
        z = torch.randn(B*num_goals, 4, latent_size, latent_size, device=device)
        y = y.flatten(0, 1)
        model_kwargs = dict(y=y, x_cond=x_cond, rel_t=rel_t)      
        samples = diffusion.p_sample_loop(
                model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=progress, device=device
        )
        samples = vae.decode(samples / 0.18215).sample

        return torch.clip(samples, -1., 1.)

def generate_rollout(args, output_dir, rollout_fps, idxs, all_models, obs_image, gt_image, delta, num_cond, device):
    rollout_stride = args.input_fps // rollout_fps
    gt_image = gt_image[:, rollout_stride-1::rollout_stride]
    delta = delta.unflatten(1, (-1, rollout_stride)).sum(2)
    curr_obs = obs_image.clone().to(device)
    
    for i in range(gt_image.shape[1]):
        curr_delta = delta[:, i:i+1].to(device)
        if args.gt:
            x_pred_pixels = gt_image[:, i].clone().to(device)
        else:
            x_pred_pixels = model_forward_wrapper(all_models, curr_obs, curr_delta, rollout_stride, args.latent_size, num_cond=num_cond, num_goals=1, device=device)

        curr_obs = torch.cat((curr_obs, x_pred_pixels.unsqueeze(1)), dim=1) # append current prediction
        curr_obs = curr_obs[:, 1:] # remove first observation
        visualize_preds(output_dir, idxs, i, x_pred_pixels)

def generate_time(args, output_dir, idxs, all_models, obs_image, gt_output, delta, secs, num_cond, device):
    eval_timesteps = [sec*args.input_fps for sec in secs]
    for sec, timestep in zip(secs, eval_timesteps):
        curr_delta = delta[:, :timestep].sum(dim=1, keepdim=True)
        if args.gt:
            x_pred_pixels = gt_output[:, timestep-1].clone().to(device)
        else:
            x_pred_pixels = model_forward_wrapper(all_models, obs_image, curr_delta, timestep, args.latent_size, num_cond=num_cond, num_goals=1, device=device)
        visualize_preds(output_dir, idxs, sec, x_pred_pixels)

def visualize_preds(output_dir, idxs, sec, x_pred_pixels):
    for batch_idx, sample_idx in enumerate(idxs.flatten()):
        sample_idx = int(sample_idx.item())
        sample_folder = os.path.join(output_dir, f'id_{sample_idx}')
        os.makedirs(sample_folder, exist_ok=True)
        image_file = os.path.join(sample_folder, f'{sec}.png')
        save_image(image_file, x_pred_pixels[batch_idx], True)

@torch.no_grad
def main(args):
    _, _, device, _ = init_distributed()
    print(args)
    device = torch.device(device)
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    exp_eval = args.exp

    # model & config setup
    if args.gt:
        args.save_output_dir = os.path.join(args.output_dir, 'gt')
    else:
        exp_name = os.path.basename(exp_eval).split('.')[0]
        args.save_output_dir = os.path.join(args.output_dir, exp_name)
    
    if  args.ckp != '0100000':
        args.save_output_dir = args.save_output_dir + "_%s"%(args.ckp)

    os.makedirs(args.save_output_dir, exist_ok=True)

    with open("config/eval_config.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config

    with open(exp_eval, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)

    latent_size = config['image_size'] // 8
    args.latent_size = config['image_size'] // 8

    num_cond = config['context_size']
    print(f"Denoiser backend selected: {args.denoiser_backend}")
    print("loading")
    model_lst = (None, None, None)
    if not args.gt:
        if args.denoiser_backend == "torch":
            model = CDiT_models[config['model']](context_size=num_cond, input_size=latent_size, in_channels=4)
            checkpoint_path = resolve_checkpoint_path(config, args)
            ckp = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            print(model.load_state_dict(ckp["ema"], strict=True))
            model.eval()
            model.to(device)
            if args.torch_compile:
                model = torch.compile(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=False)
        elif args.denoiser_backend == "onnx":
            onnx_model_path = resolve_onnx_model_path(args)
            model = ONNXCDiTWrapper(onnx_model_path, args.onnx_provider)
        else:
            raise ValueError(f"Unsupported denoiser backend: {args.denoiser_backend}")
        diffusion = create_diffusion(str(250))
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
        model_lst = (model, diffusion, vae)

    # Loading Datasets
    dataset_names = args.datasets.split(',')
    datasets = {}

    for dataset_name in dataset_names:
        dataset_val = get_dataset_eval(config, dataset_name, args.eval_type, predefined_index=True)

        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)

        curr_data_loader = torch.utils.data.DataLoader(
                            dataset_val, sampler=sampler_val,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=False
                        )
        datasets[dataset_name] = curr_data_loader

    print_freq = 1
    header = 'Evaluation: '
    metric_logger = dist.MetricLogger(delimiter="  ")

    for dataset_name in dataset_names:
        dataset_save_output_dir = os.path.join(args.save_output_dir, dataset_name)
        os.makedirs(dataset_save_output_dir, exist_ok=True)
        curr_data_loader = datasets[dataset_name]
        
        for data_iter_step, (idxs, obs_image, gt_image, delta) in enumerate(metric_logger.log_every(curr_data_loader, print_freq, header)):
            if args.max_eval_batches is not None and data_iter_step >= args.max_eval_batches:
                print(f"Stopping after --max-eval-batches {args.max_eval_batches}")
                break
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                obs_image = obs_image[:, -num_cond:].to(device)
                gt_image = gt_image.to(device)
                num_cond = config["context_size"]
                if args.eval_type == 'rollout':
                    for rollout_fps in args.rollout_fps_values:
                        curr_rollout_output_dir = os.path.join(dataset_save_output_dir, f'rollout_{rollout_fps}fps')
                        os.makedirs(curr_rollout_output_dir, exist_ok=True)
                        generate_rollout(args, curr_rollout_output_dir, rollout_fps, idxs, model_lst, obs_image, gt_image, delta, num_cond, device)
                elif args.eval_type == 'time':
                    secs = np.array([2**i for i in range(0, args.num_sec_eval)])
                    curr_time_output_dir = os.path.join(dataset_save_output_dir, 'time')
                    os.makedirs(curr_time_output_dir, exist_ok=True)
                    generate_time(args, curr_time_output_dir, idxs, model_lst, obs_image, gt_image, delta, secs, num_cond, device)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--output_dir", type=str, default=None, help="output directory")
    parser.add_argument("--exp", type=str, default=None, help="experiment name")
    parser.add_argument("--ckp", type=str, default='0100000')
    parser.add_argument("--checkpoint-path", type=str, default=None, help="explicit checkpoint path")
    parser.add_argument("--num_sec_eval", type=int, default=5)
    parser.add_argument("--input_fps", type=int, default=4)
    parser.add_argument("--datasets", type=str, default=None, help="dataset name")
    parser.add_argument("--num_workers", type=int, default=8, help="num workers")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--eval_type", type=str, default=None, help="type of evaluation has to be either 'time' or 'rollout'")
    # Rollout Evaluation Args
    parser.add_argument("--rollout_fps_values", type=str, default='1,4', help="")
    parser.add_argument("--gt", type=int, default=0, help="set to 1 to produce ground truth evaluation set")
    parser.add_argument("--torch-compile", type=int, default=0, help="enable torch.compile for inference")
    parser.add_argument("--denoiser-backend", type=str, default="torch", choices=["torch", "onnx"], help="denoiser backend")
    parser.add_argument("--onnx-model-path", type=str, default=None, help="ONNX denoiser model path")
    parser.add_argument("--onnx-provider", type=str, default="CPUExecutionProvider", help="ONNX Runtime execution provider")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="optional limit for smoke evaluation batches")
    args = parser.parse_args()
    
    args.rollout_fps_values = [int(fps) for fps in args.rollout_fps_values.split(',')]
    
    main(args)
