import time
import torch
import torch.distributed as dist

def diagnose_comm_primitives(rank: int, world_size: int, dev: torch.device, recv_timeout_sec: float):
    res = {}
    if dist.is_initialized():
        try:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            print(f"[comm_rank_check] passed={rank} actual={dist.get_rank()} world={world_size}", flush=True)
        except Exception:
            pass
    try:
        t = torch.tensor([0], dtype=torch.int32, device=dev)
        if rank == 0:
            t = torch.tensor([12345], dtype=torch.int32, device=dev)
        dist.broadcast(t, 0)
        res["bcast"] = {"ok": bool(int(t.item()) == 12345), "val": int(t.item())}
    except Exception as e:
        res["bcast"] = {"ok": False, "err": str(e)}
    try:
        inp = torch.tensor([rank], dtype=torch.int32, device=dev)
        gl = [torch.zeros(1, dtype=torch.int32, device=dev) for _ in range(world_size)]
        dist.all_gather(gl, inp)
        vals = [int(x.item()) for x in gl]
        res["all_gather"] = {"ok": bool(len(vals) == world_size), "vals": vals}
    except Exception as e:
        res["all_gather"] = {"ok": False, "err": str(e)}
    try:
        s = torch.tensor([rank], dtype=torch.int32, device=dev)
        dist.all_reduce(s, op=dist.ReduceOp.SUM)
        m = torch.tensor([rank], dtype=torch.int32, device=dev)
        dist.all_reduce(m, op=dist.ReduceOp.MIN)
        res["all_reduce"] = {"ok": True, "sum": int(s.item()), "min": int(m.item())}
    except Exception as e:
        res["all_reduce"] = {"ok": False, "err": str(e)}
    try:
        _barrier_nccl_safe()
        res["barrier"] = {"ok": True}
    except Exception as e:
        res["barrier"] = {"ok": False, "err": str(e)}
    try:
        try:
            print(f"[p2p_ring_setup] backend={dist.get_backend()} device={str(dev)} rank={rank} world={world_size}", flush=True)
        except Exception:
            pass
        try:
            _barrier_nccl_safe()
        except Exception:
            pass
        src = (rank - 1 + world_size) % world_size
        dst = (rank + 1) % world_size
        cnt_recv = torch.empty((1,), dtype=torch.int32, device=dev)
        cnt_send = torch.tensor([1], dtype=torch.int32, device=dev)
        recv_t = torch.empty((2,), dtype=torch.int32, device=dev)
        send_t = torch.tensor([rank, int(time.time()*1000) % 1000000000], dtype=torch.int32, device=dev)
        print(f"[p2p_ring_begin] rank={rank} src={src} dst={dst}", flush=True)
        if (rank % 2) == 0:
            s_cnt = dist.isend(cnt_send, dst)
            r_cnt = dist.irecv(cnt_recv, src)
            ok_cnt = _wait_work_with_timeout(r_cnt, max(recv_timeout_sec, 10.0), "diag_ring_handshake", src)
            try:
                s_cnt.wait()
            except Exception:
                pass
            swork = dist.isend(send_t, dst)
            print(f"[p2p_ring_send_posted] rank={rank} dst={dst}", flush=True)
            rwork = dist.irecv(recv_t, src)
            print(f"[p2p_ring_recv_posted] rank={rank} src={src}", flush=True)
        else:
            r_cnt = dist.irecv(cnt_recv, src)
            s_cnt = dist.isend(cnt_send, dst)
            ok_cnt = _wait_work_with_timeout(r_cnt, max(recv_timeout_sec, 10.0), "diag_ring_handshake", src)
            try:
                s_cnt.wait()
            except Exception:
                pass
            rwork = dist.irecv(recv_t, src)
            print(f"[p2p_ring_recv_posted] rank={rank} src={src}", flush=True)
            swork = dist.isend(send_t, dst)
            print(f"[p2p_ring_send_posted] rank={rank} dst={dst}", flush=True)
        okr = _wait_work_with_timeout(rwork, max(recv_timeout_sec, 10.0), "diag_ring_recv", src)
        if okr:
            print(f"[p2p_ring_recv_ok] rank={rank} from={src} recv_rank={int(recv_t[0].item())}", flush=True)
        else:
            print(f"[p2p_ring_recv_timeout] rank={rank} from={src}", flush=True)
        print(f"[p2p_ring_send_wait_begin] rank={rank} to={dst}", flush=True)
        try:
            swork.wait()
        except Exception:
            pass
        print(f"[p2p_ring_send_wait_end] rank={rank} to={dst}", flush=True)
        recv_rank = int(recv_t[0].item()) if okr else None
        res["p2p_ring"] = {"ok": bool(okr), "from": src, "to": dst, "recv_rank": recv_rank}
        status_t = torch.tensor([int(rank), int(src), int(dst), int(1 if okr else 0), int(recv_rank if recv_rank is not None else -1)], dtype=torch.int32, device=dev)
        gather_list = [torch.zeros(5, dtype=torch.int32, device=dev) for _ in range(world_size)]
        dist.all_gather(gather_list, status_t)
        summary = [[int(x[0].item()), int(x[1].item()), int(x[2].item()), int(x[3].item()), int(x[4].item())] for x in gather_list]
        print(f"[p2p_ring_summary] rank={rank} entries={summary}", flush=True)
        try:
            timeouts = [(r, s, d) for (r, s, d, ok, rr) in summary if ok == 0]
            oks = [(r, s, d, rr) for (r, s, d, ok, rr) in summary if ok == 1]
            print(f"[p2p_ring_summary_parsed] timeouts={timeouts}", flush=True)
            print(f"[p2p_ring_summary_parsed] recv_ok={oks}", flush=True)
        except Exception:
            pass
    except Exception as e:
        res["p2p_ring"] = {"ok": False, "err": str(e)}
    try:
        try:
            _barrier_nccl_safe()
        except Exception:
            pass
        src = (rank - 1 + world_size) % world_size
        dst = (rank + 1) % world_size
        recv_b = torch.empty((2,), dtype=torch.int32, device=dev)
        send_b = torch.tensor([rank, int(time.time()*1000) % 1000000000], dtype=torch.int32, device=dev)
        if (rank % 2) == 0:
            print(f"[p2p_ring_blocking_begin] rank={rank} order=recv-then-send src={src} dst={dst}", flush=True)
            dist.recv(recv_b, src)
            dist.send(send_b, dst)
        else:
            print(f"[p2p_ring_blocking_begin] rank={rank} order=send-then-recv src={src} dst={dst}", flush=True)
            dist.send(send_b, dst)
            dist.recv(recv_b, src)
        res["p2p_ring_blocking"] = {"ok": True, "recv_rank": int(recv_b[0].item())}
        status_t = torch.tensor([int(rank), int(src), int(dst), int(1), int(int(recv_b[0].item()))], dtype=torch.int32, device=dev)
        gather_list = [torch.zeros(5, dtype=torch.int32, device=dev) for _ in range(world_size)]
        dist.all_gather(gather_list, status_t)
        summary = [[int(x[0].item()), int(x[1].item()), int(x[2].item()), int(x[3].item()), int(x[4].item())] for x in gather_list]
        print(f"[p2p_ring_blocking_summary] rank={rank} entries={summary}", flush=True)
    except Exception as e:
        res["p2p_ring_blocking"] = {"ok": False, "err": str(e)}
    try:
        in_s = [1] * world_size
        out_s = [1] * world_size
        inp = torch.arange(world_size, dtype=torch.int32, device=dev) + rank * 1000
        out = torch.empty(world_size, dtype=torch.int32, device=dev)
        dist.all_to_all_single(out, inp, out_s, in_s)
        res["all_to_all"] = {"ok": True, "sample": int(out[0].item())}
    except Exception as e:
        res["all_to_all"] = {"ok": False, "err": str(e)}
    print(f"[comm_diagnose] rank={rank} results={res}", flush=True)
    return res

def _wait_work_with_timeout(work, timeout_sec: float, stage: str, src_dst: int):
    start = time.time()
    while True:
        if work.is_completed():
            try:
                work.wait()
            except Exception:
                pass
            return True
        if (time.time() - start) >= timeout_sec:
            print(f"[diag_timeout] stage={stage} peer={int(src_dst)} after={timeout_sec}s", flush=True)
            return False
        time.sleep(0.01)

def probe_comm_neighbors(rank: int, partners: list[int], dev: torch.device, recv_timeout_sec: float, blocking_probe: bool = True):
    if dist.is_initialized():
        # print(" is_initialized", flush=True)
        
        try:
            rank = dist.get_rank()
            print(f"[comm_probe_begin] rank={rank} partners={partners}", flush=True)
        except Exception:
            print(f"[comm_probe_begin] rank={rank} partners={partners}", flush=True)
    else:
        print(f"[comm_probe_begin] rank={rank} partners={partners}", flush=True)
        # print(" not initialized", flush=True)
    try:
        _barrier_nccl_safe()
    except Exception:
        pass
    if blocking_probe:
        recv_counts = {r: torch.tensor([0], dtype=torch.int32, device=dev) for r in partners}
        for peer in partners:
            t = torch.tensor([1], dtype=torch.int32, device=dev)
            if int(rank) < int(peer):
                dist.send(t, peer)
                dist.recv(recv_counts[peer], peer)
            else:
                dist.recv(recv_counts[peer], peer)
                dist.send(t, peer)
        payload = torch.tensor([int(rank), int(time.time()*1000) % 1000000000], dtype=torch.int32, device=dev)
        recv_payloads = {}
        for peer in partners:
            recv_payloads[peer] = torch.empty((2,), dtype=torch.int32, device=dev)
            if int(rank) < int(peer):
                dist.send(payload, peer)
                dist.recv(recv_payloads[peer], peer)
            else:
                dist.recv(recv_payloads[peer], peer)
                dist.send(payload, peer)
        payload_ok = []
        payload_mismatch = []
        for src in partners:
            try:
                rid = int(recv_payloads[src][0].item())
                if rid == int(src):
                    payload_ok.append(src)
                else:
                    payload_mismatch.append((src, rid))
            except Exception:
                payload_mismatch.append((src, -1))
        print(f"[comm_probe_end] rank={rank} count_timeouts=[] payload_ok={payload_ok} payload_timeouts=[] payload_mismatch={payload_mismatch}", flush=True)
        try:
            mask = torch.zeros(dist.get_world_size(), dtype=torch.int32, device=dev)
            for src in payload_ok:
                if 0 <= int(src) < dist.get_world_size():
                    mask[int(src)] = 1
            gather_list = [torch.zeros(dist.get_world_size(), dtype=torch.int32, device=dev) for _ in range(dist.get_world_size())]
            dist.all_gather(gather_list, mask)
            rows = [[int(x.item()) for x in row] for row in gather_list]
            probed_mask = torch.zeros(dist.get_world_size(), dtype=torch.int32, device=dev)
            for src in partners:
                if 0 <= int(src) < dist.get_world_size():
                    probed_mask[int(src)] = 1
            probed_gather_list = [torch.zeros(dist.get_world_size(), dtype=torch.int32, device=dev) for _ in range(dist.get_world_size())]
            dist.all_gather(probed_gather_list, probed_mask)
            probed_rows = [[int(x.item()) for x in row] for row in probed_gather_list]
            print(f"[comm_probe_summary] rank={rank} probed_rows={probed_rows} matrix_rows={rows}", flush=True)
        except Exception:
            pass
        try:
            dist.barrier()
        except Exception:
            pass
        return {"count_timeouts": [], "payload_ok": payload_ok, "payload_timeouts": []}
    send_cnt_works = []
    recv_counts = {r: torch.tensor([0], dtype=torch.int32, device=dev) for r in partners}
    recv_cnt_works = {r: dist.irecv(recv_counts[r], r) for r in partners}
    for dst in partners:
        t = torch.tensor([1], dtype=torch.int32, device=dev)
        send_cnt_works.append(dist.isend(t, dst))
    timed_out_counts = []
    for r, w in recv_cnt_works.items():
        ok = _wait_work_with_timeout(w, recv_timeout_sec, "probe_count", r)
        if not ok:
            timed_out_counts.append(r)
    payload = torch.tensor([int(rank), int(time.time()*1000) % 1000000000], dtype=torch.int32, device=dev)
    send_payload_works = []
    recv_payloads = {}
    recv_payload_works = {}
    for src in partners:
        if src in timed_out_counts:
            continue
        recv_payloads[src] = torch.empty((2,), dtype=torch.int32, device=dev)
        recv_payload_works[src] = dist.irecv(recv_payloads[src], src)
    for dst in partners:
        send_payload_works.append(dist.isend(payload, dst))
    payload_ok = []
    payload_to = []
    payload_mismatch = []
    for src, w in recv_payload_works.items():
        ok = _wait_work_with_timeout(w, recv_timeout_sec, "probe_payload", src)
        if ok:
            try:
                rid = int(recv_payloads[src][0].item())
                if rid == int(src):
                    payload_ok.append(src)
                else:
                    payload_mismatch.append((src, rid))
            except Exception:
                payload_mismatch.append((src, -1))
        else:
            payload_to.append(src)
    for w in send_cnt_works:
        try:
            w.wait()
        except Exception:
            pass
    for w in send_payload_works:
        try:
            w.wait()
        except Exception:
            pass
    print(f"[comm_probe_end] rank={rank} count_timeouts={timed_out_counts} payload_ok={payload_ok} payload_timeouts={payload_to} payload_mismatch={payload_mismatch}", flush=True)
    try:
        mask = torch.zeros(dist.get_world_size(), dtype=torch.int32, device=dev)
        for src in payload_ok:
            if 0 <= int(src) < dist.get_world_size():
                mask[int(src)] = 1
        gather_list = [torch.zeros(dist.get_world_size(), dtype=torch.int32, device=dev) for _ in range(dist.get_world_size())]
        dist.all_gather(gather_list, mask)
        rows = [[int(x.item()) for x in row] for row in gather_list]
        probed_mask = torch.zeros(dist.get_world_size(), dtype=torch.int32, device=dev)
        for src in partners:
            if 0 <= int(src) < dist.get_world_size():
                probed_mask[int(src)] = 1
        probed_gather_list = [torch.zeros(dist.get_world_size(), dtype=torch.int32, device=dev) for _ in range(dist.get_world_size())]
        dist.all_gather(probed_gather_list, probed_mask)
        probed_rows = [[int(x.item()) for x in row] for row in probed_gather_list]
        print(f"[comm_probe_summary] rank={rank} probed_rows={probed_rows} matrix_rows={rows}", flush=True)
    except Exception:
        pass
    try:
        dist.barrier()
    except Exception:
        pass
    return {"count_timeouts": timed_out_counts, "payload_ok": payload_ok, "payload_timeouts": payload_to}
def _barrier_nccl_safe():
    if not dist.is_initialized():
        return
    try:
        b = str(dist.get_backend()).lower()
    except Exception:
        b = "gloo"
    if b == "nccl" and torch.cuda.is_available():
        try:
            dev_id = torch.cuda.current_device()
            dist.barrier(device_ids=[dev_id])
            return
        except Exception:
            pass
    dist.barrier()
