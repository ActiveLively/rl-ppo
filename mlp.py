import torch
import torch.nn as nn

ACTIVATIONS = {
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "gelu": nn.GELU, 
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU
}
class Block(nn.Module):
    '''
    Linear -> norm -> activation -> dropout -> maybe residual
    '''
    def __init__(self,
            in_dim: int,
            out_dim: int,
            activation: str = "relu",
            norm: str | None = None,
            dropout: float = 0.0,
            residual: bool = False
            ):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)

        # load normalization method
        if norm == "layer":
            self.norm = nn.LayerNorm(out_dim)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(out_dim)
        elif norm is None:
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unknown norm '{norm}'")

        #load activation, dropout, and checks residual
        self.act = ACTIVATIONS[activation]()    # instantiate activation func. once w/ "()"
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        if self.residual:
            if in_dim == out_dim:
                self.skip = nn.Identity()
            else:
                self.skip = nn.Linear(in_dim, out_dim, bias = False)

    def forward(self, x):
        out = self.dropout(self.act(self.norm(self.linear(x))))
        if self.residual: 
            out = out + self.skip(x)
        return out
            

class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: list[int],
            activation: str = "relu",
            dropout: float = 0.0,
            norm: str | None = None,
            residual: bool = False,
            init: str | None = None,
            gain: float = 2 ** 0.5,  # default value for ReLU, GELU, SiLU, and Leaky ReLU
            output_gain: float = 1.0
        ):
        super().__init__()

        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'")

        layers = []

        current_dim = input_dim

        # build network 
        for hidden in hidden_dim:
            layers.append(Block(current_dim, hidden, activation, norm, dropout, residual))
            current_dim = hidden  

        layers.append(nn.Linear(current_dim, output_dim))

        self.net = nn.Sequential(*layers)

        # setup init (orthogonal)
        if init not in (None, "orthogonal"):
            raise ValueError(f"Unknown init '{init}'")

        if init == "orthogonal":
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=gain)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            nn.init.orthogonal_(self.net[-1].weight, gain=output_gain)

    def forward(self, x):
        return self.net(x)
