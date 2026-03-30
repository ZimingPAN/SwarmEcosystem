import os
import numpy as np
import math
import torch
import subprocess
import json
import re
import time
import torch.distributed as dist

from RL4KMC.utils.env import EnvKeys, env_int


def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)


def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    return e**2 / 2


def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == "Discrete":
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1
    return act_shape


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c


def print_rocm_smi():
    try:
        output = subprocess.check_output(["rocm-smi"], stderr=subprocess.STDOUT).decode(
            "utf-8"
        )
        print(output, flush=True)
    except subprocess.CalledProcessError as e:
        print("rocm-smi 执行失败：")
        print(e.output.decode("utf-8"))


def monitor_amd_gpu_status(device_id=0):
    return
    try:
        result = subprocess.check_output("rocm-smi", shell=True).decode()
        dcu_id = device_id + 1
        gpu_util = "N/A"
        vram_util = "N/A"
        for line in result.splitlines():
            line = line.strip()
            if line.startswith(str(dcu_id) + " "):
                parts = re.split(r"\s+", line)
                vram_util = parts[-2]
                gpu_util = parts[-1]
                break
        allocated = torch.cuda.memory_allocated(device_id) / 1024**2
        reserved = torch.cuda.memory_reserved(device_id) / 1024**2
        print(
            f"AMD GPU[{device_id}] "
            f"DCU利用率: {gpu_util} | "
            f"VRAM: {vram_util} | "
            f"PyTorch分配: {allocated:.1f}MB | "
            f"PyTorch预留: {reserved:.1f}MB",
            flush=True,
        )
    except Exception as e:
        print(f"获取 AMD GPU 信息失败: {e}")


def safe_barrier():
    if not dist.is_initialized():
        return
    dist.barrier()  # type: ignore[union-attr]


class Timer:
    TIME_LOOP = 0
    TIME_SOLVE = 1
    TIME_UPDATE = 2
    TIME_COMM = 3
    TIME_OUTPUT = 4
    TIME_APP = 5
    TIME_WAIT = 6
    TIME_BARRIER = 7
    TIME_BORDER = 8
    TIME_PROPENSITY = 9
    TIME_KMC_SAMPLE = 10
    TIME_MISC = 11
    TIME_DELTA_T = 12

    def __init__(self, dist_module=None):
        self.array = [0.0] * 13
        self.previous_time = 0.0
        self.dist = dist_module

    def init(self):
        for i in range(len(self.array)):
            self.array[i] = 0.0

    def stamp_reset(self):
        self.previous_time = time.perf_counter()

    def stamp(self, which):
        t = time.perf_counter()
        self.array[which] += t - self.previous_time
        self.previous_time = t

    def barrier_start(self, which):
        if self.dist is not None and getattr(self.dist, "is_initialized", lambda: False)():
            try:
                safe_barrier()
            except Exception:
                try:
                    self.dist.barrier()
                except Exception:
                    pass
        self.array[which] = time.perf_counter()

    def barrier_stop(self, which):
        if self.dist is not None and getattr(self.dist, "is_initialized", lambda: False)():
            try:
                safe_barrier()
            except Exception:
                try:
                    self.dist.barrier()
                except Exception:
                    pass
        t = time.perf_counter()
        self.array[which] = t - self.array[which]

    def elapsed(self, which):
        t = time.perf_counter()
        return t - self.array[which]


def check_cuda_memory(device=None):
    dev = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[union-attr]
        if device is None
        else torch.device(device)  # type: ignore[union-attr]
    )
    if dev.type != "cuda":
        return
    torch.cuda.synchronize(dev)  # type: ignore[union-attr]
    m0 = torch.cuda.memory_allocated(dev) / (1024**2)  # type: ignore[union-attr]
    r0 = torch.cuda.memory_reserved(dev) / (1024**2)  # type: ignore[union-attr]
    print(f"[cuda] device={dev} allocated={m0:.2f}MB reserved={r0:.2f}MB", flush=True)


class SSAEvalFileLogger:
    def __init__(
        self,
        sim_dir,
        worker_id,
        temperature,
        num_v,
        num_c,
        cluster_fp,
        cu_moves_fp,
        energy_drops_fp,
        energy_series_fp,
    ):
        self.sim_dir = str(sim_dir)
        self.worker_id = int(worker_id)
        self.temperature = float(temperature)
        self.num_v = int(num_v)
        self.num_c = int(num_c)
        self.cluster_fp = str(cluster_fp)
        self.cu_moves_fp = str(cu_moves_fp)
        self.energy_drops_fp = str(energy_drops_fp)
        self.energy_series_fp = str(energy_series_fp)

    def init_files(self):
        with open(self.cluster_fp, "w") as f:
            f.write("time,cu_cv,iso_frac\n")
        with open(self.cu_moves_fp, "w") as f:
            f.write(
                "time,cu_id,cu_topk_id,from_x,from_y,from_z,to_x,to_y,to_z,vac_id,dir_idx\n"
            )
        with open(self.energy_drops_fp, "w") as f:
            f.write(
                "time,vac_id,dir_idx,delta_E,vac_x_before,vac_y_before,vac_z_before,vac_x_after,vac_y_after,vac_z_after\n"
            )
        with open(self.energy_series_fp, "w") as f:
            f.write("step,time,total_E\n")

    def write_cluster_row(self, time_s, cu_cv, iso_frac):
        with open(self.cluster_fp, "a") as f:
            f.write(f"{float(time_s)},{float(cu_cv)},{float(iso_frac)}\n")

    def write_energy_series_row(self, step, time_s, total_e):
        with open(self.energy_series_fp, "a") as f:
            f.write(f"{int(step)},{float(time_s)},{float(total_e)}\n")

    def write_energy_drop_row(
        self, time_s, vac_id, dir_idx, delta_e, vac_before, vac_after
    ):
        bx, by, bz = map(int, vac_before)
        ax, ay, az = map(int, vac_after)
        with open(self.energy_drops_fp, "a") as f:
            f.write(
                f"{float(time_s)},{int(vac_id)},{int(dir_idx)},{float(delta_e)},{bx},{by},{bz},{ax},{ay},{az}\n"
            )

    def write_cu_move_row(
        self, time_s, cu_id, cu_topk_id, cu_from, cu_to, vac_id, dir_idx
    ):
        fx, fy, fz = map(int, cu_from)
        tx, ty, tz = map(int, cu_to)
        with open(self.cu_moves_fp, "a") as f:
            f.write(
                f"{float(time_s)},{'' if cu_id is None else int(cu_id)},{'' if cu_topk_id is None else int(cu_topk_id)},"
                f"{fx},{fy},{fz},{tx},{ty},{tz},{int(vac_id)},{int(dir_idx)}\n"
            )

    def write_advancement(self, cluster_series):
        if not isinstance(cluster_series, list) or len(cluster_series) <= 1:
            return
        C0 = float(cluster_series[0]["iso_frac"])
        Cinf = float(cluster_series[-1]["iso_frac"])
        denom = (C0 - Cinf) if abs(C0 - Cinf) > 1e-12 else 1.0
        adv_series = [
            (float(s["t"]), max(0.0, min(1.0, (C0 - float(s["iso_frac"])) / denom)))
            for s in cluster_series
        ]
        adv_fp = os.path.join(
            self.sim_dir, f"advancement_V{int(self.num_v)}_C{int(self.num_c)}.csv"
        )
        with open(adv_fp, "w") as f:
            f.write("time,advancement\n")
            for tval, zeta in adv_series:
                f.write(f"{float(tval)},{float(zeta)}\n")
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xs = [p[0] for p in adv_series]
            ys = [p[1] for p in adv_series]
            plt.figure(figsize=(6, 4))
            plt.plot(xs, ys, label=f"T={int(self.temperature)}K")
            plt.xlabel("Time (s)")
            plt.ylabel("Advancement factor ζ")
            plt.ylim(0.0, 1.0)
            plt.grid(True, alpha=0.3)
            plt.legend()
            png_fp = os.path.join(
                self.sim_dir, f"advancement_V{int(self.num_v)}_C{int(self.num_c)}.png"
            )
            plt.savefig(png_fp, dpi=150)
            plt.close()
        except Exception:
            pass

    def plot_energy_series(self):
        if not os.path.isfile(self.energy_series_fp):
            return
        times = []
        totals = []
        with open(self.energy_series_fp, "r") as f:
            next(f, None)
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    tval = float(parts[1])
                    Eval = float(parts[2])
                elif len(parts) >= 2:
                    tval = float(parts[0])
                    Eval = float(parts[1])
                else:
                    continue
                times.append(tval)
                totals.append(Eval)
        if len(times) <= 1:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(6, 4))
            plt.plot(times, totals, label=f"T={int(self.temperature)}K")
            plt.xlabel("Time (s)")
            plt.ylabel("Total energy (eV)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            png_fp = os.path.join(
                self.sim_dir, f"energy_series_worker{int(self.worker_id)}.png"
            )
            plt.savefig(png_fp, dpi=150)
            plt.close()
        except Exception:
            pass


class TaskStepTimer:
    def __init__(self, output_fp, max_steps=1000):
        self.output_fp = str(output_fp)
        self.max_steps = int(max_steps)
        self._step_idx = None
        self._t0 = None
        self._checkpoints = []
        self._wrote_header = False
        self._fh = None
        self._write_count = 0
        self._flush_every = int(env_int(EnvKeys.KMC_LOG_FLUSH_EVERY, 1024, min_value=0))
        self._flush_every = int(max(0, self._flush_every))
        self._buffer_bytes = int(
            env_int(EnvKeys.KMC_LOG_BUFFER_BYTES, 256 * 1024, min_value=0)
        )
        self._buffer_bytes = int(max(0, self._buffer_bytes))
        # Keep a persistent file handle to avoid per-step open/close syscalls.
        try:
            buf = self._buffer_bytes if self._buffer_bytes > 0 else -1
            self._fh = open(self.output_fp, "w", encoding="utf-8", buffering=buf)
        except Exception:
            self._fh = None

    def close(self):
        try:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
        finally:
            self._fh = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def start_step(self, step_idx):
        step_idx = int(step_idx)
        if step_idx >= self.max_steps:
            self._step_idx = None
            self._t0 = None
            self._checkpoints = []
            return
        self._step_idx = step_idx
        self._t0 = time.perf_counter()
        self._checkpoints = []

    def mark(self, label):
        if self._step_idx is None:
            return
        self._checkpoints.append((str(label), time.perf_counter()))

    def end_step(self):
        if self._step_idx is None or self._t0 is None:
            return
        t_end = time.perf_counter()
        total = float(t_end - self._t0)

        labels = []
        dts = []
        prev = self._t0
        for lab, ts in self._checkpoints:
            labels.append(lab)
            dts.append(float(ts - prev))
            prev = ts
        labels.append("other")
        dts.append(float(t_end - prev))

        # Write header once (avoid per-step stat/getsize).
        try:
            f = self._fh
            if f is None:
                buf = self._buffer_bytes if self._buffer_bytes > 0 else -1
                f = open(self.output_fp, "a", encoding="utf-8", buffering=buf)
                self._fh = f
            if not bool(self._wrote_header):
                f.write("step,total_time," + ",".join(labels) + "\n")
                self._wrote_header = True
            f.write(
                f"{int(self._step_idx)},{total:.6f},"
                + ",".join([f"{dt:.6f}" for dt in dts])
                + "\n"
            )
            self._write_count += 1
            if self._flush_every > 0 and (self._write_count % self._flush_every) == 0:
                try:
                    f.flush()
                except Exception:
                    pass
        except Exception:
            # Best-effort: never break compute due to timing IO.
            pass


class StepTimerController:
    """统一封装 step 计时逻辑，支持开关与空实现。"""

    def __init__(self, timer: TaskStepTimer, enabled=True):
        self._timer = timer
        self.enabled = bool(enabled) and (timer is not None)

    def start_step(self, step_idx):
        if self.enabled:
            self._timer.start_step(step_idx)

    def mark(self, label):
        if self.enabled:
            self._timer.mark(label)

    def end_step(self):
        if self.enabled:
            self._timer.end_step()


class StepCSVLogger:
    def __init__(self, output_fp, columns):
        self.output_fp = str(output_fp)
        self.columns = [str(c) for c in list(columns)]
        self._fh = None
        self._write_count = 0
        self._flush_every = int(env_int(EnvKeys.KMC_LOG_FLUSH_EVERY, 1024, min_value=0))
        self._flush_every = int(max(0, self._flush_every))
        self._buffer_bytes = int(
            env_int(EnvKeys.KMC_LOG_BUFFER_BYTES, 256 * 1024, min_value=0)
        )
        self._buffer_bytes = int(max(0, self._buffer_bytes))
        try:
            buf = self._buffer_bytes if self._buffer_bytes > 0 else -1
            self._fh = open(self.output_fp, "w", encoding="utf-8", buffering=buf)
            self._fh.write("step," + ",".join(self.columns) + "\n")
        except Exception:
            self._fh = None

    def close(self):
        try:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
        finally:
            self._fh = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def write(self, step_idx, values):
        step_idx = int(step_idx)
        row = [str(step_idx)]
        for c in self.columns:
            v = values.get(c, "")
            if isinstance(v, float):
                row.append(f"{v:.6f}")
            else:
                row.append(str(v))
        try:
            f = self._fh
            if f is None:
                buf = self._buffer_bytes if self._buffer_bytes > 0 else -1
                f = open(self.output_fp, "a", encoding="utf-8", buffering=buf)
                self._fh = f
            f.write(",".join(row) + "\n")
            self._write_count += 1
            if self._flush_every > 0 and (self._write_count % self._flush_every) == 0:
                try:
                    f.flush()
                except Exception:
                    pass
        except Exception:
            # Best-effort: logging must not break training.
            pass
