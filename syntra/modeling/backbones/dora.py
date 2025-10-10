
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from syntra.modeling.backbones.image_encoder import ImageEncoder


class _DoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            weight,
            bias,
            dim,
            m_q,
            m_v,
            lora_A_q,
            lora_B_q,
            lora_A_v,
            lora_B_v
    ):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)
        self.dim = dim
        self.m_q = nn.Parameter(m_q)
        self.m_v = nn.Parameter(m_v)
        self.lora_A_q = nn.Parameter(lora_A_q)
        self.lora_B_q = nn.Parameter(lora_B_q)
        self.lora_A_v = nn.Parameter(lora_A_v)
        self.lora_B_v = nn.Parameter(lora_B_v)

    def forward(self, x):
        lora_q = torch.matmul(self.lora_A_q, self.lora_B_q)
        adapted_q = self.weight[:self.dim, :] + lora_q
        column_norm_q = adapted_q.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)
        norm_adapted_q = adapted_q / column_norm_q
        calc_weights_q = self.m_q * norm_adapted_q

        lora_v = torch.matmul(self.lora_A_v, self.lora_B_v)
        adapted_v = self.weight[-self.dim:, :] + lora_v
        column_norm_v = adapted_v.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)
        norm_adapted_v = adapted_v / column_norm_v
        calc_weights_v = self.m_v * norm_adapted_v

        new_weights = torch.cat((calc_weights_q, self.weight[self.dim:-self.dim, :], calc_weights_v), dim=0)

        return F.linear(x, new_weights, self.bias)


class DoRAWrapper(nn.ModuleList):
    """Applies weight-decomposed low-rank adaptation to a Sam model's image encoder.

    Args:
        model (nn.Module): The image encoding backbone model to be adapted (e.g., Hiera).
        r (int): The rank for the low-rank adaptation.
        dora_layer (list, optional): List of layer indices to apply DoRA. Defaults to None, which applies DoRA to all layers.
    """

    def __init__(self, model: nn.Module, r: int, dora_layer=None):

        assert r > 0
        attn_blocks = model.blocks
        self.dora_layer = list(
                range(len(attn_blocks)))  if dora_layer is None else dora_layer
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        self.ms = []

        # freeze Hiera
        for param in model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(attn_blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.dora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features
            weight = w_qkv_linear.weight.data.clone()
            bias = w_qkv_linear.bias.data.clone()
            m_q = w_qkv_linear.weight[: dim, :].norm(p=2, dim=0, keepdim=True)
            m_v = w_qkv_linear.weight[-dim:, :].norm(p=2, dim=0, keepdim=True)

            device = weight.device
            std_dev = 1 / torch.sqrt(torch.tensor(r, device=device).float())
            lora_A_q = torch.randn(dim, r, device=device) * std_dev
            lora_B_q = torch.zeros(r, dim, device=device)
            lora_A_v = torch.randn(dim, r, device=device) * std_dev
            lora_B_v = torch.zeros(r, dim, device=device)

            self.w_As.append(lora_A_q)
            self.w_Bs.append(lora_B_q)
            self.ms.append(m_q)
            self.w_As.append(lora_A_v)
            self.w_Bs.append(lora_B_v)
            self.ms.append(m_v)

            blk.attn.qkv = _DoRA_qkv(
                weight,
                bias,
                dim,
                m_q,
                m_v,
                lora_A_q,
                lora_B_q,
                lora_A_v,
                lora_B_v
            )

        super(DoRAWrapper, self).__init__(list(attn_blocks))

    def get_dora_state(self) -> None:
        lora_tensors = {}
        m_tensors = {}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self, torch.nn.DataParallel) or isinstance(self.sam, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.module.state_dict()
        else:
            state_dict = self.state_dict()
        for key, value in state_dict.items():
            if 'qkv' in key:
                if 'm_' in key:
                    m_tensors[key] = value
                elif 'lora' in key:
                    lora_tensors[key] = value

        merged_dict = {**m_tensors, **lora_tensors}
        return merged_dict

    def load_dora_state(self, state_dict: dict) -> None:

        model_dict = self.state_dict()
        model_keys = model_dict.keys()

        # load dora
        m_keys = [k for k in model_keys if 'qkv' in k and 'm_' in k]
        m_values = [state_dict[k] for k in m_keys]
        m_new_state_dict = {k: v for k, v in zip(m_keys, m_values)}
        model_dict.update(m_new_state_dict)

        lora_keys = [k for k in model_keys if 'qkv' in k and 'lora' in k]
        lora_values = [state_dict[k] for k in lora_keys]
        lora_new_state_dict = {k: v for k, v in zip(lora_keys, lora_values)}
        model_dict.update(lora_new_state_dict)

        self.load_state_dict(model_dict)


    def forward(self, batched_input, multimask_output, image_size):
        return self.sam(batched_input, multimask_output, image_size)


if __name__ == "__main__":
 pass
