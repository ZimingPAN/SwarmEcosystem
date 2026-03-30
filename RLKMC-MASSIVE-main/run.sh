#!/usr/bin/env bash

set -Ee
set -o pipefail

time_tag=$(date +%Y%m%d-%H%M%S)

ENV_NSCCSZ=${ENV_NSCCSZ:-/home/share/fengguangnan/chq/kmc/env-scripts/env-nsccsz.sh}
ENV_THXY=${ENV_THXY:-/XYFS01/sysu_xwzhang_2/chq/workspace/kmc/env-scripts/env-thxy.sh}
if [[ -f "${ENV_NSCCSZ}" ]]; then
  # shellcheck source=/dev/null
  source "${ENV_NSCCSZ}"
fi
# if [[ -f "${ENV_THXY}" ]]; then
#   # shellcheck source=/dev/null
#   source "${ENV_THXY}"
# fi
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-"${SCRIPT_DIR}"}
cd "${PROJECT_ROOT}"

# ================= 实验配置 =================

LATTICE_BASE=${LATTICE_BASE:-100}
DIST_BACKEND=${DIST_BACKEND:-mpi}
RESCALED_SIM_TIME=${RESCALED_SIM_TIME:-${RESALED_SIM_TIME:-1}}
CU_DENSITY=${CU_DENSITY:-1.34e-2}
V_DENSITY=${V_DENSITY:-2e-4}
N_RADIAL=${N_RADIAL:-128}
N_AXIAL=${N_AXIAL:-32}

NNODES=${NNODES:-4}
NUMA_PER_NODE=${NUMA_PER_NODE:-2}
NP=$((NNODES * NUMA_PER_NODE))
WORKERS_PER_RANK=${WORKERS_PER_RANK:-32}
CORES_PER_WORKER=${CORES_PER_WORKER:-1}
ENABLE_WORKER_DEBUG_LOG=${ENABLE_WORKER_DEBUG_LOG:-0}

export LOG_LEVEL=INFO
export OMP_NUM_THREADS=1

# Debuggability: ensure tracebacks show up before mpirun's generic message.
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

export WORKERS_PER_RANK CORES_PER_WORKER ENABLE_WORKER_DEBUG_LOG
export TIMING_SUMMARY_GROUP_SIZE=128

# MPI/UCX reliability knobs for multi-rank + local multiprocessing.
# Default keeps UCX for network path but avoids local posix/sysv transports
# that can fail with /proc/<pid>/fd reopen races under heavy process churn.
MPI_DISABLE_UCX_SHM=${MPI_DISABLE_UCX_SHM:-1}
MPI_FORCE_OB1=${MPI_FORCE_OB1:-0}
UCX_TLS=${UCX_TLS:-rc,tcp,self}
UCX_POSIX_USE_PROC_LINK=${UCX_POSIX_USE_PROC_LINK:-n}
UCX_WARN_UNUSED_ENV_VARS=n

if [[ "${MPI_DISABLE_UCX_SHM}" == "1" ]]; then
  export UCX_TLS UCX_POSIX_USE_PROC_LINK UCX_WARN_UNUSED_ENV_VARS
fi

# Enable per-rank pidstat/strace profiling (Linux nodes only):
#   PROFILE_TOOLS=1 PROFILE_SECONDS=15 ./run.sh
PROFILE_TOOLS=${PROFILE_TOOLS:-0}
PROFILE_SECONDS=${PROFILE_SECONDS:-15}
PROFILE_OUT_DIR=${PROFILE_OUT_DIR:-output/profile}

MPI_ARGS=(
  -np ${NP}
  --oversubscribe
  --map-by ppr:1:numa
  --bind-to numa
  # --report-bindings
  # --mca rmaps_rank_file_physical true
  # --mca grpcomm_direct_priority 100
  # --mca coll_tuned_use_dynamic_rules true
  # --mca plm_rsh_agent /usr/bin/ssh
  # -quiet
)

if [[ "${MPI_FORCE_OB1}" == "1" ]]; then
  MPI_ARGS+=(
    --mca pml ob1
    --mca btl self,tcp,vader
  )
fi

MPI_EXPORTS=(
  -x PATH
  -x LD_LIBRARY_PATH
  -x LOG_LEVEL
  -x OMP_NUM_THREADS
  -x PYTHONUNBUFFERED
  -x PYTHONFAULTHANDLER
  -x TIMING_SUMMARY_GROUP_SIZE
  -x WORKERS_PER_RANK
  -x CORES_PER_WORKER
  -x UCX_WARN_UNUSED_ENV_VARS
)

if [[ "${MPI_DISABLE_UCX_SHM}" == "1" ]]; then
  MPI_EXPORTS+=(
    -x UCX_TLS
    -x UCX_POSIX_USE_PROC_LINK
  )
fi

BENCH_ARGS=(
  --nodes ${NNODES}
  --lattice_size ${LATTICE_BASE} ${LATTICE_BASE} ${LATTICE_BASE}
  --cu_density "${CU_DENSITY}"
  --v_density "${V_DENSITY}"
  # --use_traditional_kmc
  --n_radial "${N_RADIAL}"
  --n_axial "${N_AXIAL}"
  --rescaled_sim_time "${RESCALED_SIM_TIME}"
  --bench_step 3
  --output_dir output/${time_tag}
  --output_level 1
  --workers_per_rank ${WORKERS_PER_RANK}
  --cores_per_worker ${CORES_PER_WORKER}
  --pin_policy spread
)
if [[ "${ENABLE_WORKER_DEBUG_LOG}" == "1" ]]; then
  BENCH_ARGS+=(--enable_worker_debug_log)
fi
if [[ -n "${MODEL_DIR-}" ]]; then BENCH_ARGS+=(--model_dir "${MODEL_DIR}"); fi
if [[ -n "${EMBEDDING_MODEL-}" ]]; then BENCH_ARGS+=(--embedding_model "${EMBEDDING_MODEL}"); fi

# torchrun "${DISTRIBUTED_ARGS[@]}" scripts/scalability_bench.py "${BENCH_ARGS[@]}"

echo "Running scalability benchmark with the following parameters:"
echo "  LATTICE_BASE: ${LATTICE_BASE}"
echo "  DIST_BACKEND: ${DIST_BACKEND}"
echo "  RESCALED_SIM_TIME: ${RESCALED_SIM_TIME}"
echo "  CU_DENSITY: ${CU_DENSITY}"
echo "  V_DENSITY: ${V_DENSITY}"
echo "  N_RADIAL: ${N_RADIAL}"
echo "  N_AXIAL: ${N_AXIAL}"
echo "  WORKERS_PER_RANK: ${WORKERS_PER_RANK}"
echo "  CORES_PER_WORKER: ${CORES_PER_WORKER}"
echo "  ENABLE_WORKER_DEBUG_LOG: ${ENABLE_WORKER_DEBUG_LOG}"
echo "  NP: ${NP}"

echo "executing: mpirun ${MPI_ARGS[*]} ${MPI_EXPORTS[*]} python3 main.py ${BENCH_ARGS[*]}"

if [[ "${PROFILE_TOOLS}" == "1" ]]; then
  echo "profiling enabled: PROFILE_SECONDS=${PROFILE_SECONDS} PROFILE_OUT_DIR=${PROFILE_OUT_DIR}"
  mpirun "${MPI_ARGS[@]}" "${MPI_EXPORTS[@]}" bash scripts/profile_tools.sh python3 main.py "${BENCH_ARGS[@]}"
else
  mpirun "${MPI_ARGS[@]}" "${MPI_EXPORTS[@]}" python3 main.py "${BENCH_ARGS[@]}"
fi
# mpirun --display-allocation -np 1 hostname
