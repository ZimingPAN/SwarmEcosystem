import numpy as np
import torch
import torch.nn as nn
from RL4KMC.PPO.utils.util import init, check
from RL4KMC.PPO.utils.cnn import CNNBase
from RL4KMC.PPO.utils.mlp import MLPBase
from RL4KMC.PPO.utils.rnn import RNNLayer
from RL4KMC.PPO.utils.act import ACTLayer
from RL4KMC.PPO.utils.popart import PopArt
from RL4KMC.utils.util import get_shape_from_obs_space
from torch.distributions import Categorical
from RL4KMC.PPO.utils.mymlp import ElegantMLP

class R_Actor(nn.Module):
    def __init__(self, args, action_space, device=torch.device("cuda")):
        super().__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        model_name = str(getattr(args, 'embedding_model', 'embedding')).lower()
        if model_name == 'embedding2':
            from RL4KMC.embedding.embedding2 import VacancyEmbedding as EmbedClass
        else:
            from RL4KMC.embedding import SGDNTC_Model as EmbedClass
        self.embed = EmbedClass(args, device=device)
        self.to(device)
        self.diff_k_full = None
        self.dist_k_full = None

    def get_dist(self, obs: dict):
            """
            obs: dict, 包含 'V_features_local', 'V_coords', 'CU_coords', 'h_history' 四个张量。
                这是唯一的输入参数，用于承载所有数据。
                
            return: Categorical distribution over [N * 8]
            """
            
            # --- 1. 解包、检查和设备移动 ---
            
            # 从输入的字典 obs 中取出每个张量，并立即进行 check() 和 .to(**self.tpdv) 处理。
            # 结果将存储在与键名相同的局部变量中。
            
            V_features_local = check(obs['V_features_local']).to(**self.tpdv)
            if V_features_local.numel() == 0 or V_features_local.shape[0] == 0:
                raise ValueError("V_features_local is empty; PPO actor cannot build an action distribution.")
            # V_coords = check(obs['V_coords']).to(**self.tpdv)
            # CU_coords = check(obs['CU_coords']).to(**self.tpdv)
            topk_update_info = obs.get('topk_update_info') or {}
            vid_list = topk_update_info.get('vid_list', np.empty((0,), dtype=np.int32))
            diff_k = topk_update_info.get('diff_k')
            dist_k = topk_update_info.get('dist_k')
            vacancy_count = int(V_features_local.shape[0])
            if diff_k is None or dist_k is None:
                if self.diff_k_full is None or self.dist_k_full is None:
                    k = int(getattr(self.embed, "K", getattr(self.embed, "k", 16)))
                    self.diff_k_full = torch.zeros(
                        (vacancy_count, k, 3), **self.tpdv
                    )
                    self.dist_k_full = torch.zeros(
                        (vacancy_count, k), **self.tpdv
                    )
            else:
                diff_k = check(diff_k).to(**self.tpdv)
                dist_k = check(dist_k).to(**self.tpdv)

            if self.diff_k_full == None or self.dist_k_full == None:
                self.diff_k_full = diff_k.detach().clone() if diff_k is not None else None
                self.dist_k_full = dist_k.detach().clone() if dist_k is not None else None
            else:
                next_diff_k_full = self.diff_k_full.detach().clone()
                next_dist_k_full = self.dist_k_full.detach().clone()
                for i, vid in enumerate(vid_list):
                    next_diff_k_full[vid] = diff_k[i]
                    next_dist_k_full[vid] = dist_k[i]
                self.diff_k_full = next_diff_k_full
                self.dist_k_full = next_dist_k_full

            
            # --- 2. 调用 self.embed ---
            
            # 使用四个已处理的张量作为独立参数调用 self.embed
            logits = self.embed(
                V_features_local, 
                # V_coords, 
                # CU_coords, 
                self.diff_k_full.detach().clone(),
                self.dist_k_full.detach().clone()
                
            )  # [N * 8]
            logits = logits.reshape(-1)
            
            return Categorical(logits=logits)

    # def get_dist(self, obs):
    #     """
    #     obs: [N, obs_dim]
    #     return: Categorical distribution over [N * 8]
    #     """
    #     obs = check(obs).to(**self.tpdv)
    #     logits = self.embed(obs).reshape(-1)  # [N * 8]
    #     return Categorical(logits=logits)

    def forward(self, obs, deterministic=False):
        v_features = check(obs['V_features_local']).to(**self.tpdv)
        if v_features.numel() == 0 or v_features.shape[0] == 0:
            raise ValueError("V_features_local is empty during actor.forward().")
        # 1. 提取 NN1 邻居类型
        # obs 形状: [N, 14]
        # NN1 类型是 obs 的第二维的前 8 个数
        nn1_types = v_features[:, 0:8].to(torch.int32)  # 形状: [N, 8]
        
        # 2. 创建掩码：类型 == 2 的地方为非法动作
        # V=2 是空位类型。如果邻居是空位 (V=2)，则该跳跃 (action) 是非法的。
        # illegal_mask 形状: [N, 8]，True 表示非法 (空位跳空位)
        illegal_mask = (nn1_types == 2)
        
        # 3. 将掩码展平以匹配 dist.probs 的形状
        # dist.probs 形状: [N*8]，是所有空位所有 8 个方向的概率
        flat_illegal_mask = illegal_mask.flatten() # 形状: [N*8]
        
        # 4. 创建一个被屏蔽的概率分布 (masked_probs)
        # ❗ 注意：为了进行 argmax 和 sample，我们使用一个临时张量来屏蔽概率 ❗
        
        # 使用克隆的 probs 作为基线
        dist = self.get_dist(obs)
        masked_probs = dist.probs.clone()
        
        # 将非法动作的概率设置为 0 (或一个极小的正数)
        # 当处理概率时，直接设置为 0 即可，但在 log_probs 步骤中需要特殊处理
        masked_probs[flat_illegal_mask] = 0.0
        
        # 5. 重新归一化 (Renormalization)
        # 对于 argmax 和 sample，通常需要重新归一化以确保总和为 1，
        # 但由于 dist.probs 是一维的，我们不需要显式地重新归一化。
        # 如果分布是批次的 (N, 8)，则需要按行归一化。对于扁平化的一维数组，如果总和不影响 argmax，可以跳过。
        # 但为了采样准确性，我们应该重新归一化，或者创建一个新的 Categorical 分布。
        
        # 重新归一化步骤（确保采样准确）：
        sum_probs = masked_probs.sum()
        if sum_probs.item() > 0:
            masked_probs = masked_probs / sum_probs
        else:
            legal_indices = torch.nonzero(~flat_illegal_mask, as_tuple=False).reshape(-1)
            if legal_indices.numel() > 0:
                masked_probs = torch.zeros_like(dist.probs)
                masked_probs[legal_indices] = 1.0 / legal_indices.numel()
            else:
                masked_probs = torch.full_like(dist.probs, 1.0 / max(int(dist.probs.numel()), 1))


        # 6. 根据新的概率张量创建新的分布对象（用于 sample 和 log_probs）
        # 这样做可以确保 sample() 方法是从修正后的分布中抽样的
        masked_dist = Categorical(probs=masked_probs)

        # 7. 选择动作
        if deterministic:
            # 对于确定性选择 (argmax)，我们直接找屏蔽后概率最大的索引
            # (因为屏蔽后，非法动作的概率是 0，它们不会被选中)
            action = masked_probs.argmax().item()
        else:
            # 对于探索性采样 (sample)，从新的分布中采样
            action = masked_dist.sample().item()
        
        # 8. 计算 log_probs
        # 为了避免 log(0)，我们需要在计算 log_probs 时使用一个极小值，
        # 或者更好地，屏蔽掉 log(0) 处的值。
        # 最佳实践是将 log_probs 直接设为负无穷 (-inf)
        
        # 初始 log_probs (通常是网络输出的 logit，这里我们从 dist.probs 导出)
        log_probs_full = torch.log(dist.probs + 1e-8) 
        
        # 将非法动作的 log_probs 设为 -inf (或者一个非常小的负数，例如 -1e9)
        # 这样做在计算 RL 损失时，可以保证非法动作的梯度为 0。
        log_probs = log_probs_full.clone()
        log_probs[flat_illegal_mask] = -1e9 
        
        return action, log_probs

    # def forward(self, obs, deterministic=False):
    #     """
    #     obs: [N, obs_dim]
    #     return: sampled action (int), log_prob [N, action_space(8)]
    #     """
    #     # print("FORWARD: obs: ", obs)
    #     dist = self.get_dist(obs)
    #     # print("FORWARD: dist.probs: ", dist.probs)
    #     action = dist.probs.argmax().item() if deterministic else dist.sample().item()
    #     log_probs = torch.log(dist.probs + 1e-8)  # shape: [N*8]
    #     # print("FORWARD: log_probs: ", log_probs)
    #     return action, log_probs

    def evaluate_actions(self, obs, actions):
        """
        obs: [N, obs_dim]
        actions: [B] int tensor in [0, N*8)
        return: log_probs [B], entropy (scalar)
        """
        # print("EVALUATE_ACTIONS: obs: ", obs)
        dist = self.get_dist(obs)
        actions = check(actions).to(**self.tpdv)
        if actions.dim() > 0 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        # print("EVALUATE_ACTIONS: log_probs: ", torch.log(dist.probs + 1e-8) )
        return log_probs, entropy

class R_Critic(nn.Module):
    def __init__(self, args, device=torch.device("cuda")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        # zrg
        self.critic = ElegantMLP(
            # input_dim=9 + (args.topk * 4 + 14) * args.lattice_v_nums,
            input_dim=9 + 14 * args.lattice_v_nums,
            output_dim=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=args.layer_N,
            activation='relu',  # 也可以改为 gelu
            use_layernorm=True,
            use_orthogonal=True
        )

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        # if self._use_popart:
        # self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        # else:
        self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, sys_obs):
        sys_obs = check(sys_obs).to(**self.tpdv)
        x = self.critic(sys_obs)
        values = self.v_out(x)
        return values
