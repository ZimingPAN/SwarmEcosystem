# 世界模型实施计划

## 当前状态（2026-03-26 更新）

### 已达成里程碑
- ✅ RLKMC baseline、LightZero KMC adapter、Dreamer4 KMC adapter 全部接入
- ✅ 16 个 BCC 配位壳层 defect graph 落地
- ✅ 真实 reward 链路修复，CUDA GPU 训练启用
- ✅ **完整在线 RL 训练闭环**：MuZero (MCTS+GNN) 和 Dreamer (Actor-Critic+GNN) 独立脚本
- ✅ **MuZero v5 eval mean_reward = +20.48**（远大于 0）
- ✅ **Dreamer v4 eval mean_reward = +0.49**（scale=1 下比 PPO 好 3.7×）
- ✅ PPO baseline 验证完成：entropy 恒定 5.336，策略未学习，avg reward +0.13
- ✅ Dreamer 训练稳定性修复：优势归一化 + 策略梯度裁剪

### 关键修复
- pydeps 中 torch 2.11+cu130 覆盖系统 torch → CUDA 不可用（已修复）
- BatchNorm + batch_size=1 → LayerNorm
- 602-bin categorical → scalar regression
- Dreamer policy gradient explosion → advantage normalization + clipping

## 当前设计约束

- **不破坏 PPO baseline**：PPO 继续走 RLKMC 原始观测与训练流
- world-model 路径使用 defect graph codec / encoder
- reward 评估使用原始 reward（delta_E * reward_scale），不做正值映射
- **执行约束**：训练/测试只在 A100 服务器上运行，本地只做代码修改

## 正在运行

| 任务 | tmux 会话 | 配置 | 当前进度 |
| --- | --- | --- | --- |
| MuZero v5 | mz5 | scale=10, cu=8%, ep=100, 32 MCTS sims | 44/500 iter, eval=+20.48 |
| Dreamer v5 | dm5 | scale=10, cu=8%, ep=100 | 60/500 iter, eval=+1.69 |
| PPO baseline | ppo_baseline | scale=1, cu=5%, ep=50 | 97 episodes, 已完成 |

## 下一步

1. 等待 v5 训练完成（~500 iter, 约 20 小时），获取最终收敛 eval
2. 用相同配置 (scale=10, cu=8%, ep=100) 重新跑 PPO baseline 做公平对比
3. 2NN vs 4NN 对比实验
4. Δt 接入 MCTS 搜索目标（max -ΔE/Δt）
5. 尝试 max_defects=384, max_vacancies=32 的完整容量配置
6. 整理最终对比报告，准备 NeurIPS 投稿材料
