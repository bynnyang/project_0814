from collections import deque

import numpy as np


class ReplayMemory(object):
    def __init__(self, memory_size: int, extra_items: list = []):
        self.items = ["state", "action", "reward", "done"] + extra_items
        self.memory = {}
        for item in self.items:
            self.memory[item] = deque([], maxlen=memory_size)
    
    def push(self, observations:tuple):
        """Save a transition"""
        for i, item in enumerate(self.items):
            self.memory[item].append(observations[i])

    def get_items(self, idx_list: np.ndarray):
        batches = {}
        for item in self.items:
            batches[item] = []
        batches["next_state"] = []
        for idx in idx_list:
            for item in self.items:
                batches[item].append(self.memory[item][idx])
            if idx == self.__len__()-1 or self.memory["done"][idx]:
                batches["next_state"].append(None)
            else:
                batches["next_state"].append(self.memory["state"][idx+1])
        for idx in batches.keys():
            if isinstance(batches[idx][0], np.ndarray):
                batches[idx] = np.array(batches[idx])
        return batches

    def sample(self, batch_size: int):
        idx_list = np.random.randint(self.__len__(), size=batch_size)
        return self.get_items(idx_list)

    def shuffle(self, idx_range: int = None):
        idx_range = self.__len__() if idx_range is None else idx_range
        idx_list = np.arange(idx_range)
        np.random.shuffle(idx_list)
        return self.get_items(idx_list)

    def clear(self):
        for item in self.items:
            self.memory[item].clear()

    def __len__(self):
        return len(self.memory["state"])

class DualReplayMemory(object):
    """
    主 buffer + 成功 buffer
    - 所有 transition 进入 main
    - 成功 episode 的 transition 也进入 success
    - sample 时按 success_ratio 混合采样
    """
    def __init__(self, memory_size: int, success_memory_size: int, extra_items=None, success_ratio: float = 0.4):
        import numpy as np
        extra_items = [] if extra_items is None else list(extra_items)
        self.main = ReplayMemory(memory_size, extra_items=extra_items)
        self.success = ReplayMemory(success_memory_size, extra_items=extra_items)
        self.success_ratio = float(success_ratio)
        self._np = np

    def push(self, observations: tuple):
        self.main.push(observations)

    def push_success(self, observations: tuple):
        self.success.push(observations)

    def clear(self):
        self.main.clear()
        self.success.clear()

    def __len__(self):
        return len(self.main)

    @staticmethod
    def _merge_batches(b1, b2):
        if b1 is None:
            return b2
        if b2 is None:
            return b1
        out = {}
        keys = set(b1.keys()) | set(b2.keys())
        for k in keys:
            v1 = b1.get(k, [])
            v2 = b2.get(k, [])
            # numpy array -> concat, else -> list extend
            if hasattr(v1, "shape") and hasattr(v2, "shape"):
                out[k] = __import__("numpy").concatenate([v1, v2], axis=0)
            else:
                if hasattr(v1, "shape"):
                    v1 = list(v1)
                if hasattr(v2, "shape"):
                    v2 = list(v2)
                out[k] = list(v1) + list(v2)
        return out

    def sample(self, batch_size: int):
        # success buffer 为空或 ratio=0 -> 全部走 main
        if len(self.success) == 0 or self.success_ratio <= 0.0:
            return self.main.sample(batch_size)

        n_succ = int(round(batch_size * self.success_ratio))
        n_succ = max(1, min(n_succ, batch_size))
        n_succ = min(n_succ, len(self.success))
        n_main = batch_size - n_succ

        b_succ = self.success.sample(n_succ) if n_succ > 0 else None
        b_main = self.main.sample(n_main) if n_main > 0 else None
        return self._merge_batches(b_succ, b_main)

    def shuffle(self, idx_range: int = None):
        # 需要 shuffle 的话一般只用 main 就够了
        return self.main.shuffle(idx_range=idx_range)