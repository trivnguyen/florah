
from typing import Optional, Tuple

import torch
from torch import Tensor


class GRUDCell(torch.nn.Module):
    r"""
    A gated recurrent unit (GRU) cell with a decaying hidden state.
    Implemented based on the paper:
    "Recurrent Neural Networks for Multivariate Time Series with Missing Values"

    Hidden state is decayed by the following equation:

    .. math::
        \begin{array}{ll}
        \gamma_h = \exp{-\mathrm{max}(0, W_\gamma \delta_t + \beta_\gamma)}
        h' = \gamma_h * h
        \end{array}

    where :math:`h` is the previous hidden state.

    NOTE:
    Only hidden state decay is implemented because for our application,
    we do not need to implement input decay.
    """
    def __init__(
        self, input_size: int, hidden_size: int, delta_size: int) -> None:
        """
        Parameters
        ----------
        input_size: int
            feature size of input tensor
        hidden_size: int
            hidden size of GRU cells
        delta_size: int
            feature size of time step difference
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.delta_size = delta_size

        # GRU cell
        self.gru_cell = torch.nn.GRUCell(input_size, hidden_size)

        # linear layer for hidden state decay
        self.linear_gamma_h = torch.nn.Linear(delta_size, hidden_size)

    def forward(
            self, x: Tensor, delta: Tensor,
            h: Optional[Tensor] = None) -> Tensor:
        # init hidden state to zeros if not given
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_size,
                            dtype=x.dtype, device=x.device)

        gamma_h = torch.exp(
            -torch.max(
                torch.zeros(h.size(), dtype=x.dtype, device=x.device),
                self.linear_gamma_h(delta)
            ))
        h = gamma_h * h
        h = self.gru_cell(x, h)
        return h

class GRUD(torch.nn.Module):
    r"""
    A gated recurrent unit (GRU) cell with a decaying hidden state.
    Implemented based on the paper:
    "Recurrent Neural Networks for Multivariate Time Series with Missing Values"

    Hidden state is decayed by the following equation:

    .. math::
        \begin{array}{ll}
        \gamma_h = \exp{-\mathrm{max}(0, W_\gamma \delta_t + \beta_\gamma)}
        h' = \gamma_h * h
        \end{array}

    where :math:`h` is the previous hidden state.

    NOTE:
    Only hidden state decay is implemented because for our application,
    we do not need to implement input decay.
    """
    def __init__(
        self, input_size: int, hidden_size: int, delta_size: int,
        batch_first: bool = False) -> None:
        """
        Parameters
        ----------
        input_size: int
            feature size of input tensor
        hidden_size: int
            hidden size of GRU cells
        delta_size: int
            feature size of time step difference
        """

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.delta_size = delta_size
        self.batch_first = batch_first
        self.grud_cell = GRUDCell(input_size, hidden_size, delta_size)

    def forward(
            self, x: Tensor, delta: Tensor, h: Optional[Tensor] = None
        ) -> Tuple[(Tensor, Tensor)]:
        # convert tensor to contiguous if needed
        if self.batch_first:
            x_cont = x.transpose(0, 1).contiguous()
            delta_cont = delta.transpose(0, 1).contiguous()
        else:
            x_cont = x
            delta_cont = delta

        # init hidden state if not given
        if h is None:
            h = torch.zeros(x_cont.size(1), self.hidden_size,
                            dtype=x.dtype, device=x.device)

        # iterate over x
        x_out = []
        for i in range(len(x_cont)):
            h = self.grud_cell(x_cont[i], delta_cont[i], h)
            x_out.append(h)
        x_out = torch.stack(x_out)

        if self.batch_first:
            x_out = x_out.transpose(0, 1).contiguous()

        return x_out, h
