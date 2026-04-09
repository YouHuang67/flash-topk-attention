from __future__ import annotations

from typing import Dict


def _next_power_of_2(x: int) -> int:
    if x <= 0:
        return 1
    return 1 << (x - 1).bit_length()


def heuristic_scoring_fwd(
    N: int, SCORE_BS_ORIG: int, D: int,
) -> Dict[str, int]:
    SCORE_BS = _next_power_of_2(SCORE_BS_ORIG)
    IS_POW2 = (SCORE_BS_ORIG == SCORE_BS)

    Q_BS = 16 if N <= 512 else 32

    if IS_POW2:
        if D <= 64 and SCORE_BS >= 128:
            KV_BS = 128
        elif D <= 64:
            KV_BS = 64
        else:
            KV_BS = 32
    else:
        if D <= 64 and SCORE_BS >= 512:
            KV_BS = 128
        elif D >= 128:
            KV_BS = 32
        else:
            candidates = [128, 64, 32]
            KV_BS = 32
            for kv in candidates:
                iters = (SCORE_BS_ORIG + kv - 1) // kv
                eff = SCORE_BS_ORIG / (iters * kv)
                if eff >= 0.85:
                    KV_BS = kv
                    break

    return {"Q_BS": Q_BS, "KV_BS": KV_BS, "num_warps": 4}


def heuristic_scoring_dq(N: int, D: int) -> Dict[str, int]:
    if D <= 64:
        return {"Q_BS": 32, "KV_BS": 64, "num_warps": 4}
    elif N >= 65536:
        return {"Q_BS": 64, "KV_BS": 64, "num_warps": 8}
    else:
        return {"Q_BS": 32, "KV_BS": 32, "num_warps": 4}


def heuristic_scoring_dkv(N: int, D: int) -> Dict[str, int]:
    return {"KV_BS": 32, "Q_BS": 32, "num_warps": 4}


def heuristic_attention_fwd(
    Q_BS: int, KV_BS: int, D: int,
) -> Dict[str, int]:
    Q_TILE = 32 if Q_BS >= 32 else 16

    if D >= 128:
        KV_TILE = 32
    elif Q_BS >= 32 and KV_BS >= 128:
        KV_TILE = 64
    elif Q_BS < 32 and KV_BS >= 180:
        KV_TILE = 64
    else:
        KV_TILE = 32

    return {
        "Q_TILE": Q_TILE,
        "KV_TILE": KV_TILE,
        "num_warps": 4,
        "num_stages": 2,
    }


def heuristic_autotune(configs, key, heuristic_fn, heuristic_key_args):
    def decorator(fn):
        return _HeuristicAutotuner(
            fn, configs, key, heuristic_fn, heuristic_key_args,
        )
    return decorator


class _HeuristicAutotuner:

    def __init__(self, fn, configs, key, heuristic_fn, heuristic_key_args):
        self.fn = fn
        self.configs = configs
        self.key = key
        self.heuristic_fn = heuristic_fn
        self.heuristic_key_args = heuristic_key_args
        self.arg_names = fn.arg_names

    def __getitem__(self, grid):
        return _HeuristicLauncher(self, grid)

    def run(self, *args, **kwargs):
        raise NotImplementedError("Use kernel[grid](...) syntax")


class _HeuristicLauncher:

    def __init__(self, autotuner, grid):
        self.autotuner = autotuner
        self.grid = grid

    def __call__(self, *args, **kwargs):
        at = self.autotuner
        all_kwargs = dict(zip(at.arg_names, args))
        all_kwargs.update(kwargs)

        h_args = {k: all_kwargs[k] for k in at.heuristic_key_args}
        config = at.heuristic_fn(**h_args)

        special_keys = {"num_warps", "num_stages", "num_ctas"}
        launch_kwargs = {}
        meta_kwargs = {}
        for k, v in config.items():
            if k in special_keys:
                meta_kwargs[k] = v
            else:
                launch_kwargs[k] = v

        all_kwargs.update(launch_kwargs)

        if callable(self.grid):
            grid = self.grid(all_kwargs)
        else:
            grid = self.grid

        at.fn[grid](*args, **kwargs, **launch_kwargs, **meta_kwargs)
