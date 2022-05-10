import os
import sys
import stat
import subprocess
from loguru import logger
from datetime import datetime


def add_cluster_args(parser):
    parser.add_argument("--agent_id", type=int, default=0)
    parser.add_argument("--num_exp", type=int, default=1, help="log every k steps")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--cluster_node", type=str, default="")
    parser.add_argument("--bid", type=int, default=21, help="log every k steps")
    parser.add_argument(
        "--gpu_min_mem", type=int, default=16000, help="log every k steps"
    )
    parser.add_argument("--memory", type=int, default=8000, help="log every k steps")
    parser.add_argument("--num_workers", type=int, default=8, help="log every k steps")
    parser.add_argument("--gpu_arch", type=str, default="all", help="log every k steps")
    parser.add_argument("--num_gpus", type=int, default=0, help="log every k steps")
    return parser


class CondorCluster:
    GPUS = {
        "v100-p16": ('"Tesla V100-PCIE-16GB"', "tesla", 16000),
        "v100-p32": ('"Tesla V100-PCIE-32GB"', "tesla", 32000),
        "v100-s32": ('"Tesla V100-SXM2-32GB"', "tesla", 32000),
        "quadro6000": ('"Quadro RTX 6000"', "quadro", 24000),
        "rtx2080ti": ('"GeForce RTX 2080 Ti"', "rtx", 11000),
        "a100-80": ('"NVIDIA A100-SXM-80GB"', "ampere", 80000),
        "a100-40": ('"NVIDIA A100-SXM4-40GB"', "ampere", 40000),
    }

    def __init__(
        self,
        args,
        script,
        num_exp=1,
    ):
        """
        :param script: (str) python script which will be executed eg. "main.py"
        :param cfg: (yacs.config.CfgNode) CfgNode object
        :param cfg_file: (str) path to yaml config file eg. config.yaml
        """
        self.script = script
        self.num_exp = num_exp
        self.cfg = args

        # if isinstance(cfg, list):
        #     self.num_exp = len(cfg)
        #     cfg = cfg[0]

        self.logs_folder = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    def submit(self):
        gpus = self._get_gpus(min_mem=self.cfg.gpu_min_mem, arch=self.cfg.gpu_arch)
        gpus = " || ".join([f"CUDADeviceName=={x}" for x in gpus])

        os.makedirs("run_scripts", exist_ok=True)
        run_script = os.path.join("run_scripts", f"run_{self.logs_folder}.sh")

        self._create_submission_file(run_script, gpus)
        logger.info(
            f"The logs for this experiments can be found under: condor_logs/{self.logs_folder}"
        )

        self._create_bash_file(run_script)

        cmd = ["condor_submit_bid", f"{self.cfg.bid}", "submit.sub"]
        logger.info("Executing " + " ".join(cmd))
        subprocess.call(cmd)

    def _create_submission_file(self, run_script, gpus):
        os.makedirs(os.path.join("condor_logs", self.logs_folder), exist_ok=True)
        submission = (
            f"executable = {run_script}\n"
            "arguments = $(Process) $(Cluster)\n"
            f"error = condor_logs/{self.logs_folder}/$(Cluster).$(Process).err\n"
            f"output = condor_logs/{self.logs_folder}/$(Cluster).$(Process).out\n"
            f"log = condor_logs/{self.logs_folder}/$(Cluster).$(Process).log\n"
            f"request_memory = {self.cfg.memory}\n"
            f"request_cpus={int(self.cfg.num_workers / 2)}\n"
            f"request_gpus={self.cfg.num_gpus}\n"
            f"requirements={gpus}\n"
            f"queue {self.num_exp}"
        )
        # f'next_job_start_delay=10\n' \

        with open("submit.sub", "w") as f:
            f.write(submission)

    def _create_bash_file(self, run_script):
        api_key = os.environ["AOHMR_API_KEY"]
        workspace = os.environ["AOHMR_WORKSPACE"]
        mano_path_r = os.environ["MANO_MODEL_DIR_R"]
        mano_path_l = os.environ["MANO_MODEL_DIR_L"]
        bash = "export PYTHONBUFFERED=1\n"
        bash += "export PATH=$PATH\n"
        bash += f'export AOHMR_API_KEY="{api_key}"\n'
        bash += f'export AOHMR_WORKSPACE="{workspace}"\n'
        bash += f'export MANO_MODEL_DIR_R="{mano_path_r}"\n'
        bash += f'export MANO_MODEL_DIR_L="{mano_path_l}"\n'
        bash += f'export MANO_MODEL_DIR_L="{mano_path_l}"\n'
        bash += 'export EMAIL_ACC="hijason78@gmail.com"\n'
        bash += 'export EMAIL_PASS="denote.foist.finnish.seasick"\n'
        bash += f"{sys.executable} {self.script}"
        bash += " --agent_id $1"
        bash += " --cluster_node $2.$1"

        with open(run_script, "w") as f:
            f.write(bash)

        os.chmod(run_script, stat.S_IRWXU)

    def _get_gpus(self, min_mem, arch):
        if arch == "all":
            arch = ["tesla", "quadro", "rtx"]

        gpu_names = []
        for k, (gpu_name, gpu_arch, gpu_mem) in self.GPUS.items():
            if gpu_mem >= min_mem and gpu_arch in arch:
                gpu_names.append(gpu_name)

        assert len(gpu_names) > 0, "Suitable GPU model could not be found"

        return gpu_names
