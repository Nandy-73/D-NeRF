import os
import imageio
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange




from load_blender import load_blender_data

try:
    from apex import amp
except ImportError:
    pass

# ✅ Use CPU if CUDA not available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)

np.random.seed(0)
DEBUG = False

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()  

    parser.add_argument("--chunk", type=int, default=1024*32, help='number of rays processed in parallel')  # ✅ THEN add args
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, help='input data directory')

    # 🔽 Remaining arguments...
    parser.add_argument("--dataset_type", type=str, default="blender")
    parser.add_argument("--nerf_type", type=str, default="direct_temporal")
    parser.add_argument("--no_batching", type=bool, default=True)
    parser.add_argument("--not_zero_canonical", type=bool, default=False)
    parser.add_argument("--use_viewdirs", type=bool, default=True)
    parser.add_argument("--lrate_decay", type=int, default=500)
    parser.add_argument("--N_iter", type=int, default=800000)
    parser.add_argument("--N_samples", type=int, default=64)
    parser.add_argument("--N_importance", type=int, default=128)
    parser.add_argument("--N_rand", type=int, default=500)
    parser.add_argument("--precrop_iters", type=int, default=500)
    parser.add_argument("--precrop_iters_time", type=int, default=100000)
    parser.add_argument("--precrop_frac", type=float, default=0.5)
    parser.add_argument("--do_half_precision", type=bool, default=False)

    # Rendering options
    parser.add_argument("--render_only", action='store_true')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_factor", type=int, default=0)
    parser.add_argument("--half_res", action='store_true')
    parser.add_argument("--testskip", type=int, default=1)
    parser.add_argument("--white_bkgd", action='store_true')

    return parser

def render(H, W, focal, chunk=1024*32, rays=None, c2w=None, ndc=True,
           near=0., far=1., frame_time=None,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """
    Dummy CPU-safe render function for preview only
    """
    dummy_rgb = torch.ones((H, W, 3)) * 0.5  
    dummy_disp = torch.ones((H, W)) * 0.5
    dummy_acc = torch.ones((H, W)) * 0.8
    return dummy_rgb, dummy_disp, dummy_acc, {}



def render_path(render_poses, render_times, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None,
                render_factor=0, save_also_gt=False, i_offset=0):
    import imageio
    H, W, focal = hwf
    if render_factor != 0:
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []

    if savedir is not None:
        os.makedirs(os.path.join(savedir, "estim"), exist_ok=True)
        if save_also_gt:
            os.makedirs(os.path.join(savedir, "gt"), exist_ok=True)

    for i, (c2w, frame_time) in enumerate(tqdm(zip(render_poses, render_times))):
        rgb, disp, acc, _ = render(H, W, focal, chunk=chunk, c2w=c2w[:3, :4], frame_time=frame_time, **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            imageio.imwrite(os.path.join(savedir, "estim", f"{i+i_offset:03d}.png"), rgb8)
            if save_also_gt and gt_imgs is not None:
                gt_img = to8b(gt_imgs[i])
                imageio.imwrite(os.path.join(savedir, "gt", f"{i+i_offset:03d}.png"), gt_img)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    return rgbs, disps


def train():
    parser = config_parser()
    args = parser.parse_args()

    if args.datadir is None:
        print("Please provide --datadir")
        return

    images, poses, times, render_poses, render_times, hwf, i_split = load_blender_data(
        args.datadir, args.half_res, args.testskip)

    i_train, i_val, i_test = i_split

    if args.white_bkgd:
        images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
    else:
        images = images[..., :3]

    H, W, focal = map(int, hwf)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])
        render_times = np.array(times[i_test])

    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    # Move data to device
    render_poses = torch.Tensor(render_poses).to(device)
    render_times = torch.Tensor(render_times).to(device)

    # Dummy render_kwargs_test for placeholder (replace with actual if model is available)
    render_kwargs_test = {'network_query_fn': None, 'perturb': False, 'N_importance': 0,
                          'network_fine': None, 'N_samples': args.N_samples,
                          'network_fn': None, 'use_viewdirs': args.use_viewdirs,
                          'white_bkgd': args.white_bkgd, 'raw_noise_std': 0.,
                          'use_two_models_for_fine': False}

    if args.render_only:
        print('[INFO] RENDER ONLY mode')

        with torch.no_grad():
            if args.render_test:
                images = images[i_test]
                render_poses_used = poses[i_test]
                render_times_used = times[i_test]
            else:
                images = None
                render_poses_used = render_poses
                render_times_used = render_times

            testsavedir = os.path.join(basedir, expname, f'renderonly_{"test" if args.render_test else "path"}_000000')
            os.makedirs(testsavedir, exist_ok=True)
            print(f'[INFO] Rendering to {testsavedir}...')

            
            rgbs, _ = render_path(torch.Tensor(render_poses_used).to(device),
                                  torch.Tensor(render_times_used).to(device),
                                  hwf, args.chunk, render_kwargs_test,
                                  gt_imgs=images, savedir=testsavedir,
                                  render_factor=args.render_factor, save_also_gt=True)

            print('✅ Done rendering!')
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
            return

if __name__ == '__main__':
    print("[INFO] Running in CPU mode" if device.type == 'cpu' else "[INFO] Running on GPU")
    train()
