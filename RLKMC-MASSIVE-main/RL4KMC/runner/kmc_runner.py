import time
import os
import numpy as np
from itertools import chain
import torch

from RL4KMC.utils.util import update_linear_schedule
from RL4KMC.runner.base_runner import Runner
import imageio
from RL4KMC.envs.kmc_env import KMCEnvWrap
from tqdm import tqdm

def _t2n(x):
    return x.detach().cpu().numpy()

class KMCRunner(Runner):
    def __init__(self, args, config):
        super(KMCRunner, self).__init__(config)
        # self.envs = KMCEnvWrap(args)
        self.envs = config["envs"]

        self.reward_history = []
        self.energy_history = []
        self.output_energy_filename = "output-rl.e"

    def write_single_step_e(self, nstep, current_time, avg_time, energy, is_first_step=False):
            """
            将单个时间步的数据以指定格式追加写入到 output-rl.e 文件中。
            
            参数:
                nstep: 当前步数（从0开始）
                current_time: 瞬时时间 (time(s))
                avg_time: 平均时间 (<time>(s))
                energy: 总能量 (Etotal(eV))
                is_first_step: 是否为第一步 (nstep=0)，用于写入头部和处理时间格式
            """
            filename = self.output_energy_filename  # 使用 __init__ 中定义的属性
            
            # 模式 'a' 表示追加写入。如果是第一步，我们使用 'w' 重新创建文件并写入头部。
            mode = 'w' if is_first_step else 'a'
            
            with open(filename, mode) as f:
                if is_first_step:
                    # 写入头部注释行
                    f.write("# nstep   time(s)  <time>(s)   Etotal(eV)\n")
                
                # 格式化输出
                if nstep == 0:
                    # nstep=0 时，时间不使用科学计数法
                    line = f"{nstep:4d}  {current_time:>8}  {avg_time:>8}     {energy:.3f}\n"
                else:
                    # 时间使用科学计数法，保留5位小数；能量保留3位小数
                    line = f"{nstep:4d}  {current_time:.5e}  {avg_time:.5e}     {energy:.3f}\n"
                f.write(line)


    def run(self):

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            self.reward_history = []
            self.energy_history = []            
            self.warmup()   
            for step in tqdm(range(self.episode_length), desc="Collocting Data"):
                values, actions, action_log_probs = self.collect(step)
                obs, share_obs, positions, rewards, dones, info = self.envs.step(actions, episode)
                self.reward_history.append(np.mean(rewards))
                self.energy_history.append(info['energy_change'])
                data = obs, share_obs, positions, rewards, dones, info, values, actions, action_log_probs
                self.insert(data)

            # self.plot_reward_energy_curve(episode)

            print("Start Training...")
            self.compute()
            train_infos = self.train()
            print("Training Done...")
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                # Calculate key metrics
                fps = int(total_num_steps / (end - start))
                avg_rewards = np.mean(self.buffer.rewards) * self.episode_length
                avg_value_loss = np.mean(train_infos['value_loss'])
                avg_policy_loss = np.mean(train_infos['policy_loss'])
                avg_entropy = np.mean(train_infos['dist_entropy'])
                avg_advantage_mean = np.mean(train_infos['mean_advantages'])
                avg_advantage_std = np.mean(train_infos['std_advantages'])
                
                # Get current system energy from env
                current_energy = self.envs.env.calculate_system_energy()

                print("\n" + "="*60)
                print(f" Episode    : {episode:<5d} / {episodes:<5d}    |  Steps: {total_num_steps:<8d} / {self.num_env_steps}")
                print(f" FPS        : {fps:<10d}       |  Time Elapsed: {end - start:.2f}s")
                print(f" AvgReward  : {avg_rewards:<10.4f} |  Entropy     : {avg_entropy:.4f}")
                print(f" Value Loss : {avg_value_loss:<10.4f} |  Policy Loss : {avg_policy_loss:.4f}")
                print(f" Advantage  : {avg_advantage_mean:<10.4f} ± {avg_advantage_std:<10.4f}") 
                print(f" Energy     : {current_energy:<10.4f}")
                print(f" Scenario   : {self.all_args.scenario_name}")
                print(f" Algorithm  : {self.algorithm_name}")
                print(f" Experiment : {self.experiment_name}")
                print("="*60 + "\n")

                # Log to wandb if enabled
                if self.use_wandb:
                    wandb.log({
                        "episode": episode,
                        "total_steps": total_num_steps,
                        "fps": fps,
                        "average_reward": avg_rewards,
                        "value_loss": avg_value_loss,
                        "policy_loss": avg_policy_loss,
                        "policy_entropy": avg_entropy,
                        "system_energy": current_energy,
                        "time_elapsed": end-start,
                        "advantages": avg_advantage_mean,
                        "advantages_std": avg_advantage_std,
                    },step=total_num_steps)
            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                print("Start Evaluating...")
                self.eval(total_num_steps, episode)
                print("Evaluating Done...")
        pass


    def warmup(self):
        obs, share_obs = self.envs.reset()
        self.buffer.share_obs[0] = share_obs
        self.buffer.obs[0] = obs

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        share_obs_batch = self.buffer.share_obs[step]
        obs_batch = self.buffer.obs[step]
        values, actions, action_log_probs = self.trainer.policy.get_actions(share_obs_batch, obs_batch)
        return values, actions, action_log_probs

    def insert(self, data):
        (obs, share_obs, positions, rewards, dones, infos, values, actions, action_log_probs) = data
        self.buffer.insert(
            share_obs=share_obs,
            obs=obs,
            positions=positions,
            actions=actions,
            action_log_probs=action_log_probs.cpu().numpy(),
            value_preds=values.cpu().numpy(),
            rewards=rewards,
        )

    def generate_output_e(self, time_history, avg_time_history, energy_history, filename="output_rl.e"):
        """
        根据时间历史、平均时间历史和能量历史生成指定格式的output.e文件
        
        参数:
            time_history: 瞬时时间数据列表/数组 (time(s))
            avg_time_history: 平均时间数据列表/数组 (<time>(s))
            energy_history: 总能量数据列表/数组 (Etotal(eV))
            filename: 输出文件名，默认为"output.e"
        """
        
        # # 检查输入数据长度是否一致
        # if not (len(time_history) == len(avg_time_history) == len(energy_history)):
        #     raise ValueError("时间、平均时间和能量历史的长度必须一致")

        # 转换为numpy数组便于处理
        time_array = np.array(time_history)
        avg_time_array = np.array(avg_time_history)
        energy_array = np.array(energy_history)
        

        # 打开文件写入
        with open(filename, 'w') as f:
            # 写入头部注释行
            f.write("# nstep   time(s)  <time>(s)   Etotal(eV)\n")
            
            # 写入数据行（nstep从0开始）
            for nstep, (t, avg_t, energy) in enumerate(zip(time_array, avg_time_array, energy_array)):
                # 格式化输出，匹配示例格式
                # 对于nstep=0的特殊情况单独处理（时间为0时不使用科学计数法）
                if nstep == 0:
                    line = f"{nstep:4d}  {t:>8}  {avg_t:>8}     {energy:.3f}\n"
                else:
                    # 时间使用科学计数法，保留5位小数；能量保留3位小数
                    line = f"{nstep:4d}  {t:.5e}  {avg_t:.5e}     {energy:.3f}\n"
                f.write(line)
        
        print(f"已生成指定格式的output.e文件: {filename}")

    @torch.no_grad()
    def eval(self, total_num_steps, episode):
        eval_episode_rewards = []
        eval_obs, _ = self.eval_envs.reset()

        # for eval_step in range(self.episode_length):
        #     print("eval_step: ", eval_step)
        #     self.trainer.prep_rollout()
        #     eval_actions, _ = self.trainer.policy.act(eval_obs)
        #     eval_obs, _, _, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions, episode)
        #     eval_episode_rewards.append(eval_rewards)

        import time  # 确保导入time模块

        for eval_step in range(self.episode_length):
            print("eval_step: ", eval_step)
            # self.trainer.prep_rollout()
            
            # 记录act方法的执行时间
            start_act = time.time()
            eval_actions, _ = self.trainer.policy.act(eval_obs)
            time_act = time.time() - start_act
            
            # 记录step方法的执行时间
            start_step = time.time()
            if eval_step % 2000 == 0:
                eval_obs, _, _, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions, episode)
            else:
                eval_obs = self.eval_envs.env.step_only_jump(eval_actions, episode)
                
            time_step = time.time() - start_step
            
            # 输出计时结果
            print(f"  act()用时: {time_act:.6f}秒")
            print(f"  step()用时: {time_step:.6f}秒")
            
            eval_episode_rewards.append(eval_rewards)
            if eval_step % 2000 == 0:
                self.write_single_step_e(eval_step, self.eval_envs.env.time_history[-1], self.eval_envs.env.time_history[-1], self.eval_envs.env.energy_history[-1], eval_step == 0)
            
        eval_episode_rewards = np.array(eval_episode_rewards)

        eval_train_infos = []

        eval_average_episode_rewards = np.sum(eval_episode_rewards)
        eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards})

        self.log_train(eval_train_infos, total_num_steps)  
        # print(self.eval_envs.env.time_history, self.eval_envs.env.time_history, self.eval_envs.env.energy_history)
        # self.generate_output_e(self.eval_envs.env.time_history, self.eval_envs.env.time_history, self.eval_envs.env.energy_history, "output-rl.e")

    @torch.no_grad()
    def render(self):        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render('rgb_array')[0][0]
                all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()
                
                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(np.array(list(obs[:, agent_id])),
                                                                        rnn_states[:, agent_id],
                                                                        masks[:, agent_id],
                                                                        deterministic=True)

                    action = action.detach().cpu().numpy()
                    # rearrange action
                    if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)
                   
                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = self.envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            episode_rewards = np.array(episode_rewards)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))
        
        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
