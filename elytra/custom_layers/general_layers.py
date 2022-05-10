import torch.nn as nn
from ..torch_utils import get_activation


class MLP(nn.Module):
    """
    A multi-layer perceptron with dim `dims` with activation between layers.
    """

    def __init__(self, dims, act, use_bn=False):
        super().__init__()
        assert isinstance(dims, list)
        assert isinstance(act, str)
        assert len(dims) > 1
        self.dims = dims
        self.use_bn = use_bn

        layer_list = []
        prev_dim = dims[0]
        dims = dims[1:]
        for i, curr_dim in enumerate(dims):
            if use_bn:
                layer_list.append(nn.BatchNorm1d(prev_dim))
            layer_list.append(nn.Linear(prev_dim, curr_dim))
            if i < len(dims) - 1:
                myact = get_activation("act")
                layer_list.append(myact)
            prev_dim = curr_dim
        self.model = nn.ModuleList(layer_list)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x
