"""Bounded CPU preparation with RNG committed only after each consumed batch."""

import copy
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def case_batch(config):
    value = config["training"].get("caseBatch", 1)
    return value[config["variant"]] if isinstance(value, dict) else value


def prepared_batches(order, offset, size, random, make, max_round=2, workers=0):
    planned = np.random.default_rng()
    planned.bit_generator.state = copy.deepcopy(random.bit_generator.state)

    def plan():
        for number in range(offset, len(order), size):
            ids = order[number : number + size]
            rounds = [int(planned.integers(max_round + 1)) for _ in ids]
            state = copy.deepcopy(planned.bit_generator.state)
            yield number, list(zip(ids, rounds, strict=True)), state

    def prepare(items):
        return [make(index, round) for index, round in items]

    if not workers:
        for number, items, state in plan():
            yield number, prepare(items), state
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        waiting, source = deque(), iter(plan())
        for _ in range(workers + 1):
            item = next(source, None)
            if item:
                number, items, state = item
                waiting.append((number, executor.submit(prepare, items), state))
        while waiting:
            number, future, state = waiting.popleft()
            yield number, future.result(), state
            item = next(source, None)
            if item:
                number, items, state = item
                waiting.append((number, executor.submit(prepare, items), state))
