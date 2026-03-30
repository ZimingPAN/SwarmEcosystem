import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import GATConv
# from torch_geometric.nn import radius_graph
# from torch_geometric.data import Data
from RL4KMC.PPO.utils.util import init, check
from RL4KMC.PPO.utils.mymlp import ElegantMLP

class VacancyEmbedding(nn.Module):
    def __init__(self, args, device=torch.device("cuda")):
        super().__init__()
        self.args = args
        self.device = device
        self.hidden_size = 256
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.local_encoder = ElegantMLP(
            input_dim=14,
            output_dim=8,
            hidden_size=256,
            num_layers=5,
            activation='relu',  # 也可以改为 gelu
            use_layernorm=True,
            use_orthogonal=True
        )

        # self.gnn_layers = nn.ModuleList([
        #     GATConv(
        #         in_channels=self.hidden_size,
        #         out_channels=self.hidden_size // args.gnn_attention_heads,
        #         heads=args.gnn_attention_heads,
        #         concat=True,
        #         dropout=args.gnn_attention_dropout,
        #         edge_dim=4
        #     ) for _ in range(args.gnn_layers)
        # ])
        # self.residual_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.to(device)

    def forward(self, V_feat: torch.Tensor, diff_k=None, dist_k=None, vv_edge_index=None) -> torch.Tensor:
        local_stats = check(V_feat).to(**self.tpdv)
        node_feat = self.local_encoder(local_stats)
        return node_feat


# class VacancyEmbedding(nn.Module):
#     def __init__(self, args, device=torch.device("cuda")):
#         super().__init__()
#         self.args = args
#         self.device = device
#         self.hidden_size = args.hidden_size
#         self.tpdv = dict(dtype=torch.float32, device=device)

#         self.local_encoder = ElegantMLP(
#             input_dim=14,
#             output_dim=8,
#             hidden_size=self.hidden_size,
#             num_layers=args.layer_N,
#             activation='relu',  # 也可以改为 gelu
#             use_layernorm=True,
#             use_orthogonal=True
#         )

#         # self.gnn_layers = nn.ModuleList([
#         #     GATConv(
#         #         in_channels=self.hidden_size,
#         #         out_channels=self.hidden_size // args.gnn_attention_heads,
#         #         heads=args.gnn_attention_heads,
#         #         concat=True,
#         #         dropout=args.gnn_attention_dropout,
#         #         edge_dim=4
#         #     ) for _ in range(args.gnn_layers)
#         # ])
#         # self.residual_proj = nn.Linear(self.hidden_size, self.hidden_size)
#         self.to(device)

#     def forward(self, V_feat: torch.Tensor, diff_k=None, dist_k=None, vv_edge_index=None) -> torch.Tensor:
#         local_stats = check(V_feat).to(**self.tpdv)
#         node_feat = self.local_encoder(local_stats)
#         return node_feat
