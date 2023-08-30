import traceback
import torch
import pytorch_lightning as pl
import elytra.pl_utils as pl_utils
import torch.optim as optim
from elytra.comet_utils import log_dict
from elytra.pl_utils import push_checkpoint_metric, avg_losses_cpu
import time
import json
import numpy as np
from pprint import pprint
from elytra.unidict import unidict
from loguru import logger
import os.path as op
import os
from src.utils.const import experiment, email_client

OK_CODE, NAN_LOSS_ERR, HIGH_LOSS_ERR, *_ = range(20)


def detect_loss_anomaly(loss_dict, max_val, step, warmup_steps):
    for key, val in loss_dict.items():
        if torch.isnan(val).sum() > 0:
            return NAN_LOSS_ERR, f"{key} contains NaN values"
        if val.mean() > max_val and step > warmup_steps:
            return HIGH_LOSS_ERR, f"{key} contains high loss value {val.mean()}"
    return OK_CODE, ""


def report_err(msg, batch, exp_key, failed_state_p):
    my_err_msg = f"{msg}\nFailed state at {failed_state_p}"
    logger.error(my_err_msg)
    torch.save(batch, failed_state_p)
    email_client.notify(f"Aborting {exp_key}", my_err_msg)
    raise Exception(my_err_msg)


class AbstractPL(pl.LightningModule):
    def __init__(
        self,
        args,
        module_name,
        get_model_pl_fn,
        visualize_all_fn,
        push_images_fn,
        tracked_metric,
        metric_init_val,
        high_loss_val,
        warmup_steps,
    ):
        super().__init__()
        self.module_name = module_name
        self.experiment = None
        self.args = args
        self.tracked_metric = tracked_metric
        self.metric_init_val = metric_init_val

        self.model = get_model_pl_fn(args)
        self.started_training = False
        self.loss_dict_vec = []
        self.has_applied_decay = False
        self.visualize_all = visualize_all_fn
        self.push_images = push_images_fn
        self.vis_train_batches = []
        self.vis_val_batches = []
        self.failed_state_p = op.join("logs", self.args.exp_key, "failed_state.pt")
        self.high_loss_val = high_loss_val
        self.warmup_steps = warmup_steps

    def set_training_flags(self):
        self.started_training = True

    def load_from_ckpt(self, ckpt_path):
        sd = torch.load(ckpt_path)["state_dict"]
        print(self.load_state_dict(sd))

    def training_step(self, batch, batch_idx):
        self.set_training_flags()
        if len(self.vis_train_batches) < 1:
            self.vis_train_batches.append(batch)
        inputs, targets, meta_info = batch

        try:
            loss = self.model(inputs, targets, meta_info, "train")
        except Exception as e:
            msg = traceback.format_exc()
            report_err(msg, batch, self.args.exp_key, self.failed_state_p)

        err_code, err_msg = detect_loss_anomaly(
            loss, self.high_loss_val, self.global_step, self.warmup_steps
        )
        if err_code != OK_CODE:
            report_err(err_msg, batch, self.args.exp_key, self.failed_state_p)

        loss = {k: loss[k].mean().view(-1) for k in loss}
        total_loss = sum(loss[k] for k in loss)

        loss_dict = {"total_loss": total_loss, "loss": total_loss}
        loss_dict.update(loss)

        for k, v in loss_dict.items():
            if k != "loss":
                loss_dict[k] = v.detach()

        log_every = self.args.log_every
        self.loss_dict_vec.append(loss_dict)
        self.loss_dict_vec = self.loss_dict_vec[len(self.loss_dict_vec) - log_every :]
        if batch_idx % log_every == 0 and batch_idx != 0:
            running_loss_dict = avg_losses_cpu(self.loss_dict_vec)
            running_loss_dict = unidict(running_loss_dict).postfix("__train")
            log_dict(experiment, running_loss_dict, step=self.global_step)

        return loss_dict

    def training_epoch_end(self, outputs):
        outputs = avg_losses_cpu(outputs)
        experiment.log_epoch_end(self.current_epoch)

    def validation_step(self, batch, batch_idx):
        if len(self.vis_val_batches) < 2:
            self.vis_val_batches.append(batch)
        out = self.inference_step(batch, batch_idx)
        return out

    def validation_epoch_end(self, outputs):
        return self.inference_epoch_end(outputs, postfix="__val")

    def test_step(self, batch, batch_idx):
        out = self.inference_step(batch, batch_idx)
        return out

    def test_epoch_end(self, outputs):
        """
        Test is called by trainer.test()
        if self.interface_p is None: only does evaluation on either the given dataloader
        else: dump the evaluation results to the interface_p
        """
        result, metrics, metric_dict = self.inference_epoch_end(
            outputs, postfix="__test"
        )
        for k, v in metric_dict.items():
            metric_dict[k] = float(v)

        # dump image names
        if self.args.interface_p is not None:
            imgnames = result["interface.meta_info.imgname"]
            json_p = op.join(
                op.dirname(self.args.interface_p),
                op.basename(self.args.interface_p).split(".")[0] + ".imgname.json",
            )
            with open(json_p, "w") as fp:
                json.dump({"imgname": imgnames}, fp, indent=4)

            torch.save(result, self.args.interface_p, pickle_protocol=4)

            print(f"Results: {self.args.interface_p}")

        """
        if self.metric_p is not None:
            torch.save(metrics, self.metric_p)
            json_p = self.metric_p.replace(".pt", ".json")
            with open(json_p, "w") as f:
                json.dump(metric_dict, f, indent=4)
            print(f"Metrics: {self.metric_p}")
            print(f"Metric dict: {json_p}")
        """

        return result

    def inference_step(self, batch, batch_idx):
        if self.training:
            self.eval()
        with torch.no_grad():
            inputs, targets, meta_info = batch
            out, loss = self.model(inputs, targets, meta_info, "test")
            return {"out_dict": out, "loss": loss}

    def inference_epoch_end(self, out_list, postfix):
        if not self.started_training:
            self.started_training = True
            result = push_checkpoint_metric(self.tracked_metric, self.metric_init_val)
            return result

        # unpack
        outputs, loss_dict = pl_utils.reform_outputs(out_list)

        if "test" in postfix:
            per_img_metric_dict = {}
            for k, v in outputs.items():
                if "metric." in k:
                    per_img_metric_dict[k] = np.array(v)

        metric_dict = {}
        for k, v in outputs.items():
            if "metric." in k:
                metric_dict[k] = np.nanmean(np.array(v))

        loss_metric_dict = {}
        loss_metric_dict.update(metric_dict)
        loss_metric_dict.update(loss_dict)
        loss_metric_dict = unidict(loss_metric_dict).postfix(postfix)

        log_dict(
            experiment,
            loss_metric_dict,
            step=self.global_step,
        )

        if self.args.interface_p is None:
            result = push_checkpoint_metric(
                self.tracked_metric, loss_metric_dict[self.tracked_metric]
            )
            self.log(self.tracked_metric, result[self.tracked_metric])

        if not self.args.no_vis:
            print("Rendering train images")
            self.visualize_batches(self.vis_train_batches, "_train", 2, None)
            print("Rendering val images")
            self.visualize_batches(self.vis_val_batches, "_val", 2, None)

        if "test" in postfix:
            return (
                outputs,
                {"per_img_metric_dict": per_img_metric_dict},
                metric_dict,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, self.args.lr_dec_epoch, gamma=self.args.lr_decay, verbose=True
        )
        return [optimizer], [scheduler]

    def visualize_batches(
        self, batches, postfix, num_examples, no_tqdm=True, dump_vis=False
    ):
        im_list = []
        if self.training:
            self.eval()

        tic = time.time()
        for batch_idx, batch in enumerate(batches):
            with torch.no_grad():
                inputs, targets, meta_info = batch
                vis_dict = self.model(inputs, targets, meta_info, "vis")
                curr_im_list = self.visualize_all(
                    vis_dict,
                    num_examples,
                    self.model.renderer,
                    postfix=postfix,
                    no_tqdm=no_tqdm,
                )
                im_list += curr_im_list
                print("Rendering: %d/%d" % (batch_idx + 1, len(batches)))

        if dump_vis:
            for curr_im in im_list:
                out_p = op.join(
                    "demo", curr_im["fig_name"].replace("__rend_demo", ".png")
                )
                out_folder = op.dirname(out_p)
                os.makedirs(out_folder, exist_ok=True)
                print(out_p)
                curr_im["im"].save(out_p)

        self.push_images(experiment, im_list, self.global_step)
        print("Done rendering (%.1fs)" % (time.time() - tic))
        return im_list
