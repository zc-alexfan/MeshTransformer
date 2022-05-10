import comet_ml
from tqdm import tqdm
import torch
import os
import numpy as np
import torch
import time
from loguru import logger
import os.path as op
import elytra.sys_utils as sys_utils


def init_experiment(args):
    api_key = os.environ["AOHMR_API_KEY"]
    workspace = os.environ["AOHMR_WORKSPACE"]
    args.git_commit = sys_utils.get_commit_hash()
    args.git_branch = sys_utils.get_branch()
    project_name = args.project
    disabled = args.mute
    is_new = args.resume_ckpt == ""

    if is_new:
        experiment = comet_ml.Experiment(
            api_key=api_key,
            workspace=workspace,
            project_name=project_name,
            disabled=disabled,
            display_summary_level=0,
        )
        exp_key = fetch_key_from_experiment(experiment)

        tags = ["git:" + args.git_commit[:8], args.git_branch, args.parser_type]
        logger.info(f"Experiment tags: {tags}")
        experiment.add_tags(tags)
    else:
        meta = torch.load(
            op.join("/".join(args.resume_ckpt.split("/")[:-2]), "meta.pt")
        )
        prev_key = meta["comet_key"]
        experiment = comet_ml.ExistingExperiment(
            previous_experiment=prev_key,
            api_key=api_key,
            project_name=project_name,
            workspace=workspace,
            disabled=disabled,
            display_summary_level=0,
        )
        exp_key = prev_key[:9]
    args.exp_key = exp_key
    args.log_dir = f"./logs/{args.exp_key}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.add(
        os.path.join(args.log_dir, "train.log"),
        level="INFO",
        colorize=True,
    )
    logger.info(torch.cuda.get_device_properties(device))
    args.gpu = torch.cuda.get_device_properties(device).name

    if is_new:
        os.makedirs(args.log_dir, exist_ok=True)
        meta_info = {"comet_key": experiment.get_key()}
        torch.save(meta_info, op.join(args.log_dir, "meta.pt"))
    return experiment, args


def log_dict(experiment, metric_dict, step, postfix=None):
    if experiment is None:
        return
    for key, value in metric_dict.items():
        if postfix is not None:
            key = key + postfix
        if isinstance(value, torch.Tensor) and len(value.view(-1)) == 1:
            value = value.item()

        if isinstance(value, (int, float, np.float32)):
            experiment.log_metric(key, value, step=step)


def fetch_key_from_experiment(experiment):
    if experiment is not None:
        key = str(experiment.get_key())
        key = key[:9]
        experiment.set_name(key)
    else:
        import random

        hash = random.getrandbits(128)
        key = "%032x" % (hash)
        key = key[:9]
    return key


def push_images(experiment, all_im_list, global_step=None, no_tqdm=False, verbose=True):
    if verbose:
        print("Pushing PIL images")
        tic = time.time()
    iterator = all_im_list if no_tqdm else tqdm(all_im_list)
    for im in iterator:
        im_np = np.array(im["im"])
        if "fig_name" in im.keys():
            experiment.log_image(im_np, im["fig_name"], step=global_step)
        else:
            experiment.log_image(im_np, "unnamed", step=global_step)
    if verbose:
        toc = time.time()
        print("Done pushing PIL images (%.1fs)" % (toc - tic))
