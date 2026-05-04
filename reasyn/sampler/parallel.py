# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing as mp
mp.set_start_method('spawn', force=True)
import pathlib
import subprocess
from multiprocessing import synchronize as sync
from typing import TypeAlias
import time

import pandas as pd
from tqdm.auto import tqdm

from reasyn.sampler.sampler import Sampler
from reasyn.chem.mol import Molecule
from reasyn.sampler.runtime import PathLike, load_sampling_runtime, set_process_affinity_to_all_cpus
from reasyn.utils.sample_utils import TimeLimit

import warnings
warnings.filterwarnings('ignore')


TaskQueueType: TypeAlias = "mp.JoinableQueue[Molecule | None]"
ResultQueueType: TypeAlias = "mp.Queue[tuple[Molecule, pd.DataFrame]]"


class Worker(mp.Process):
    def __init__(
        self,
        model_path: pathlib.Path | list[pathlib.Path],
        task_queue: TaskQueueType,
        result_queue: ResultQueueType,
        gpu_id: str,
        gpu_lock: sync.Lock,
        sampler_opt: dict | None = None,
        max_evolve_steps: int = 8,
        max_results: int = 100,
        time_limit: int = 1000,
        add_bb_path: str = None,
        verbose=True,
        exact_break=True,
        num_cycles: int = 1,
        num_editflow_samples: int = 10,
        num_editflow_steps: int = 100,
        mols_to_filter = None,
        filter_sim: float = 0.8,
        asset_dir: PathLike | None = None,
    ):
        super().__init__()
        self._model_path = model_path
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._gpu_id = gpu_id
        self._gpu_lock = gpu_lock

        self._sampler_opt = sampler_opt or {}
        self._sampler_opt['exact_break'] = exact_break
        self._max_evolve_steps = max_evolve_steps
        self._max_results = max_results
        self._time_limit = time_limit
        self.add_bb_path = add_bb_path
        self.verbose = verbose
        self.num_cycles = num_cycles
        self.num_editflow_samples = num_editflow_samples
        self.num_editflow_steps = num_editflow_steps
        self.mols_to_filter = mols_to_filter
        self.filter_sim = filter_sim
        self.asset_dir = asset_dir

    def run(self) -> None:
        set_process_affinity_to_all_cpus()
        
        assert isinstance(self._model_path, list) and len(self._model_path) == 2

        runtime = load_sampling_runtime(
            self._model_path,
            device=f"cuda:{self._gpu_id}",
            asset_dir=self.asset_dir,
            add_bb_path=self.add_bb_path,
            verbose=self.verbose,
        )
        self._model = runtime.models
        self._fpindex = runtime.fpindex
        self._rxn_matrix = runtime.rxn_matrix

        while True:
            next_task = self._task_queue.get()
            if next_task is None:
                self._task_queue.task_done()
                break
            try:
                result_df = self.process(next_task)
                self._task_queue.task_done()
                self._result_queue.put((next_task, result_df))
            except KeyboardInterrupt:
                print(f"{self.name}: Exiting due to KeyboardInterrupt")
                return
            
    def process(self, mol: Molecule):
        sampler = Sampler(
            fpindex=self._fpindex,
            rxn_matrix=self._rxn_matrix,
            mol=mol,
            model=self._model,
            **self._sampler_opt,
        )

        tl = TimeLimit(self._time_limit)
        t_start = time.time()
        try:
            sampler.evolve(gpu_lock=self._gpu_lock, time_limit=tl,
                            num_cycles=self.num_cycles,
                            max_evolve_steps=self._max_evolve_steps,
                            num_editflow_samples=self.num_editflow_samples,
                            num_editflow_steps=self.num_editflow_steps)
            t_elapsed = time.time() - t_start
            df = sampler.get_dataframe()[: self._max_results]
            df['time'] = t_elapsed
            return df
            
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        
        except Exception as e:
            print(f'{mol.csmiles}')
            print(e)
        

class WorkerPool:
    def __init__(
        self,
        gpu_ids: list[int | str],
        num_workers_per_gpu: int,
        task_qsize: int,
        result_qsize: int,
        **worker_opt,
    ) -> None:
        super().__init__()
        self._task_queue: TaskQueueType = mp.JoinableQueue(task_qsize)
        self._result_queue: ResultQueueType = mp.Queue(result_qsize)
        self._gpu_ids = [str(d) for d in gpu_ids]
        self._gpu_locks = [mp.Lock() for _ in gpu_ids]
        num_gpus = len(gpu_ids)
        num_workers = num_workers_per_gpu * num_gpus
        self._workers = [
            Worker(
                task_queue=self._task_queue,
                result_queue=self._result_queue,
                gpu_id=self._gpu_ids[i % num_gpus],
                gpu_lock=self._gpu_locks[i % num_gpus],
                **worker_opt,
            )
            for i in range(num_workers)
        ]

        for w in self._workers:
            w.start()

    def submit(self, task: Molecule, block: bool = True, timeout: float | None = None):
        self._task_queue.put(task, block=block, timeout=timeout)

    def fetch(self, block: bool = True, timeout: float | None = None):
        return self._result_queue.get(block=block, timeout=timeout)

    def kill(self):
        for w in self._workers:
            w.kill()
        self._result_queue.close()
        self._task_queue.close()

    def end(self):
        for _ in self._workers:
            self._task_queue.put(None)
        self._task_queue.join()
        for w in self._workers:
            w.terminate()
        self._result_queue.close()
        self._task_queue.close()


def _count_gpus():
    return int(
        subprocess.check_output(
            "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l", shell=True, text=True
        ).strip()
    )


def run_parallel_sampling(
    input: list[Molecule],
    output: pathlib.Path,
    model_path: pathlib.Path | list[pathlib.Path],
    search_width: int = 24,
    exhaustiveness: int = 64,
    num_gpus: int = -1,
    num_workers_per_gpu: int = 8,
    task_qsize: int = 0,
    result_qsize: int = 0,
    time_limit: int = 1000,
    add_bb_path: str = None,
    exact_break: bool = True,
    num_cycles: int = 1,
    num_editflow_samples: int = 10,
    num_editflow_steps: int = 100,
    mols_to_filter = None,
    filter_sim: float = 0.8,
    asset_dir: PathLike | None = None,
) -> None:
    num_gpus = num_gpus if num_gpus > 0 else _count_gpus()
    
    pool = WorkerPool(
        gpu_ids=list(range(num_gpus)),
        num_workers_per_gpu=num_workers_per_gpu,
        task_qsize=task_qsize,
        result_qsize=result_qsize,
        model_path=model_path,
        sampler_opt={
            "factor": search_width,
            "max_active_states": exhaustiveness,
            "mols_to_filter": mols_to_filter,
            "filter_sim": filter_sim
        },
        time_limit=time_limit,
        add_bb_path=add_bb_path,
        exact_break=exact_break,
        num_cycles=num_cycles,
        num_editflow_samples=num_editflow_samples,
        num_editflow_steps=num_editflow_steps,
        mols_to_filter=mols_to_filter,
        filter_sim=filter_sim,
        asset_dir=asset_dir,
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    total = len(input)
    for mol in input:
        pool.submit(mol)

    df_all: list[pd.DataFrame] = []
    with open(output, "w") as f:
        for _ in tqdm(range(total)):
            _, df = pool.fetch()
            if df is None or len(df) == 0:
                continue
            df.to_csv(f, float_format="%.3f", index=False, header=f.tell() == 0)
            df_all.append(df)

    if not df_all:
        msg = "Success rate: 0\n"
    else:
        df_merge = pd.concat(df_all, ignore_index=True)
        # canonicalize
        df_merge['target'] = df_merge['target'].apply(lambda s: Molecule(s).csmiles)
        df_merge['smiles'] = df_merge['smiles'].apply(lambda s: Molecule(s).csmiles)
        df_merge = df_merge.drop_duplicates()
        with open(output, "w") as f:
            df_merge.to_csv(f, float_format="%.3f", index=False)
        
        msg = f'{df_merge.loc[df_merge.groupby("target").idxmax()["score"]].select_dtypes(include="number").sum() / total}\n'
        count_success = len(df_merge["target"].unique())
        msg += f"Success rate: {count_success}/{total} = {count_success / total:.3f}\n"
        df_merge = df_merge[df_merge['target'] == df_merge['smiles']]
        df_merge = df_merge.drop_duplicates('target')
        count_recons = len(df_merge)
        msg += f"Reconstruction rate: {count_recons}/{total} = {count_recons / total:.3f}\n"
    print(msg)
    pool.end()


def run_parallel_sampling_return_smiles(
    input: list[Molecule],
    model_path: pathlib.Path | list[pathlib.Path],
    search_width: int = 24,
    exhaustiveness: int = 64,
    num_gpus: int = -1,
    num_workers_per_gpu: int = 8,
    task_qsize: int = 0,
    result_qsize: int = 0,
    time_limit: int = 1000,
    add_bb_path: str = None,
    num_cycles: int = 1,
    num_editflow_samples: int = 10,
    num_editflow_steps: int = 100,
    mols_to_filter=None,
    filter_sim: float = 0.8,
    asset_dir: PathLike | None = None,
) -> None:
    num_gpus = num_gpus if num_gpus > 0 else _count_gpus()
    
    pool = WorkerPool(
        gpu_ids=list(range(num_gpus)),
        num_workers_per_gpu=num_workers_per_gpu,
        task_qsize=task_qsize,
        result_qsize=result_qsize,
        model_path=model_path,
        sampler_opt={
            "factor": search_width,
            "max_active_states": exhaustiveness,
            "mols_to_filter": mols_to_filter,
            "filter_sim": filter_sim
        },
        time_limit=time_limit,
        add_bb_path=add_bb_path,
        verbose=False,
        exact_break=True,
        num_cycles=num_cycles,
        num_editflow_samples=num_editflow_samples,
        num_editflow_steps=num_editflow_steps,
        asset_dir=asset_dir,
    )

    total = len(input)
    for mol in input:
        pool.submit(mol)

    df_all: list[pd.DataFrame] = []

    for _ in tqdm(range(total)):
        _, df = pool.fetch()
        if df is None or len(df) == 0:
            continue
        df_all.append(df)

    df_merge = pd.concat(df_all, ignore_index=True)
    pool.end()

    return df_merge


def run_sampling_one(
    input: Molecule,
    model_path: str | pathlib.Path | list[pathlib.Path],
    search_width: int = 24,
    exhaustiveness: int = 64,
    max_evolve_steps: int = 8,
    max_results: int = 100,
    time_limit: int = 1000,
    add_bb_path: pathlib.Path = None,
    device='cuda',
    num_cycles: int = 1,
    num_editflow_samples: int = 10,
    num_editflow_steps: int = 100,
    mols_to_filter=None,
    filter_sim=0.8,
    asset_dir: PathLike | None = None,
) -> pd.DataFrame:
    runtime = load_sampling_runtime(
        model_path,
        device=device,
        asset_dir=asset_dir,
        add_bb_path=add_bb_path,
    )
    model = runtime.models
    
    sampler_opt={
        "factor": search_width,
        "max_active_states": exhaustiveness,
        "mols_to_filter": mols_to_filter,
        "filter_sim": filter_sim
    }

    sampler = Sampler(
        fpindex=runtime.fpindex,
        rxn_matrix=runtime.rxn_matrix,
        mol=input,
        model=model,
        **sampler_opt,
    )
    tl = TimeLimit(time_limit)
    t_start = time.time()
    sampler.evolve(gpu_lock=None, time_limit=tl,
                    num_cycles=num_cycles,
                    max_evolve_steps=max_evolve_steps,
                    num_editflow_samples=num_editflow_samples,
                    num_editflow_steps=num_editflow_steps)
    t_elapsed = time.time() - t_start
    df = sampler.get_dataframe()[: max_results]
    df['time'] = t_elapsed
    return df
