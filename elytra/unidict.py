import torch
import elytra.torch_utils as torch_utils
from loguru import logger


class unidict(dict):
    def __init__(self, mydict=None):
        if mydict is None:
            return

        for k, v in mydict.items():
            self[k] = v

    def register(self, k, v):
        assert k not in self.keys()
        self[k] = v

    def overwrite(self, k, v):
        assert k in self.keys()
        self[k] = v

    def add(self, k, v):
        self[k] = v

    def remove(self, k):
        assert k in self.keys()
        self.pop(k, None)

    def merge(self, dict2):
        assert isinstance(dict2, (dict, unidict))
        mykeys = set(self.keys())
        intersect = mykeys.intersection(set(dict2.keys()))
        assert len(intersect) == 0, f"Merge failed: duplicate keys ({intersect})"
        self.update(dict2)

    def prefix(self, text):
        out_dict = {}
        for k in self.keys():
            out_dict[text + k] = self[k]
        return unidict(out_dict)

    def replace_keys(self, str_src, str_tar):
        out_dict = {}
        for k in self.keys():
            old_key = k
            new_key = old_key.replace(str_src, str_tar)
            out_dict[new_key] = self[k]
        return unidict(out_dict)

    def postfix(self, text):
        out_dict = {}
        for k in self.keys():
            out_dict[k + text] = self[k]
        return unidict(out_dict)

    def sorted_keys(self):
        return sorted(list(self.keys()))

    def to(self, dev):
        return unidict(torch_utils.dict2dev(self, dev))

    def to_torch(self):
        return unidict(torch_utils.dict2torch(self))

    def to_np(self):
        return unidict(torch_utils.dict2np(self))

    def detach(self):
        out = {}
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().detach()
            out[k] = v
        return unidict(out)

    def check_invalid(self):
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                if torch.isnan(v).any():
                    logger.warning(f"{k} contains nan values")
                if torch.isinf(v).any():
                    logger.warning(f"{k} contains inf values")
