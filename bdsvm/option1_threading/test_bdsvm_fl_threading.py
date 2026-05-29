#!/usr/bin/env python3
"""Threaded local federated simulation for POM1 DSVM/BDSVM on MNIST."""

from __future__ import annotations

import copy
import dataclasses
import queue
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from MMLL.nodes.MasterNode import MasterNode
from MMLL.nodes.WorkerNode import WorkerNode


class TimedOutException(Exception):
    """Compatibility timeout exception matching pycloudmessenger checks."""


TimedOutException.__module__ = 'pycloudmessenger.ffl.fflapi'


@dataclasses.dataclass
class MockReceivedPacket:
    """Packet container with the same attributes used by the original comms layer."""

    content: dict
    notification: Dict[str, str]


class NullLogger:
    """Minimal logger interface used by MMLL objects."""

    def info(self, message: str) -> None:
        return


class ThreadedMessageBus:
    """In-memory message transport for one master and multiple workers."""

    def __init__(self, worker_ids: Sequence[str]):
        self._master_queue: queue.Queue[Tuple[str, dict]] = queue.Queue()
        self._worker_queues: Dict[str, queue.Queue[dict]] = {
            worker_id: queue.Queue() for worker_id in worker_ids
        }

    def send_to_master(self, sender: str, message: dict) -> None:
        self._master_queue.put((sender, copy.deepcopy(message)))

    def send_to_worker(self, worker_id: str, message: dict) -> None:
        self._worker_queues[worker_id].put(copy.deepcopy(message))

    def recv_for_master(self, timeout: float) -> MockReceivedPacket:
        try:
            sender, message = self._master_queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimedOutException('Timed out waiting for worker message') from exc
        return MockReceivedPacket(content=message, notification={'participant': sender})

    def recv_for_worker(self, worker_id: str, timeout: float) -> MockReceivedPacket:
        try:
            message = self._worker_queues[worker_id].get(timeout=timeout)
        except queue.Empty as exc:
            raise TimedOutException('Timed out waiting for master message') from exc
        return MockReceivedPacket(content=message, notification={'participant': 'ma'})


class MockComms_master:
    """Queue-backed drop-in replacement for POM1 master comms."""

    def __init__(self, bus: ThreadedMessageBus, worker_ids: Sequence[str], logger: Optional[NullLogger] = None):
        self.name = 'pycloudmessenger'
        self.workers_ids = list(worker_ids)
        self.logger = logger
        self._bus = bus

    def send(self, message: dict, destiny: str) -> None:
        self._bus.send_to_worker(destiny, message)

    def broadcast(self, message: dict, receivers_list: Optional[Sequence[str]] = None) -> None:
        receivers = self.workers_ids if receivers_list is None else receivers_list
        for worker_id in receivers:
            self._bus.send_to_worker(worker_id, message)

    def receive(self, timeout: float = 1.0) -> dict:
        packet = self._bus.recv_for_master(timeout)
        message = packet.content
        message.update({'sender': packet.notification['participant']})
        return message

    def receive_poms_123(self, timeout: float = 10.0) -> MockReceivedPacket:
        return self._bus.recv_for_master(timeout)


class MockComms_worker:
    """Queue-backed drop-in replacement for POM1 worker comms."""

    def __init__(self, bus: ThreadedMessageBus, worker_id: str, logger: Optional[NullLogger] = None):
        self.id = worker_id
        self.name = 'pycloudmessenger'
        self.logger = logger
        self._bus = bus

    def send(self, message: dict, address: Optional[str] = None) -> None:
        self._bus.send_to_master(self.id, message)

    def receive(self, timeout: float = 0.1) -> dict:
        return self._bus.recv_for_worker(self.id, timeout).content

    def receive_poms_123(self, timeout: float = 10.0) -> MockReceivedPacket:
        return self._bus.recv_for_worker(self.id, timeout)


@dataclasses.dataclass
class SimulationConfig:
    n_workers: int = 3
    nc: int = 50
    nmaxiter: int = 12
    tolerance: float = 1e-4
    sigma: float = 6.0
    c_value: float = 1.0
    eps: float = 1e-5
    random_state: int = 42


@dataclasses.dataclass
class SimulationResult:
    model: object
    fit_seconds: float
    worker_threads: List[threading.Thread]


def load_mnist_binary(random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST and produce train, validation, and test splits for digit 0 vs rest."""
    print('Loading MNIST from sklearn OpenML...')
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    X = X.astype(np.float32) / 255.0
    y = y.astype(np.int64)
    y_binary = np.where(y == 0, 1, -1).astype(np.int64)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y_binary,
        test_size=10000,
        random_state=random_state,
        stratify=y_binary,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=6000,
        random_state=random_state,
        stratify=y_train_full,
    )
    print(f'Train samples: {X_train.shape[0]}, validation samples: {X_val.shape[0]}, test samples: {X_test.shape[0]}')
    return X_train, y_train, X_val, y_val, X_test, y_test


def split_iid_among_workers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_workers: int,
    random_state: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create an IID stratified split across workers."""
    splitter = StratifiedKFold(n_splits=n_workers, shuffle=True, random_state=random_state)
    partitions: List[Tuple[np.ndarray, np.ndarray]] = []
    for _, worker_idx in splitter.split(X_train, y_train):
        partitions.append((X_train[worker_idx], y_train[worker_idx]))
    return partitions


def compute_metrics(model: object, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Evaluate an MMLL DSVM model on a dataset."""
    scores = np.asarray(model.predict(X)).reshape(-1)
    preds = np.where(scores >= 0, 1, -1)
    y_binary = (y == 1).astype(np.int64)
    return {
        'accuracy': accuracy_score(y, preds),
        'precision': precision_score(y, preds, pos_label=1, zero_division=0),
        'recall': recall_score(y, preds, pos_label=1, zero_division=0),
        'f1': f1_score(y, preds, pos_label=1, zero_division=0),
        'roc_auc': roc_auc_score(y_binary, scores),
    }


def print_metrics(label: str, metrics: Dict[str, float], fit_seconds: float) -> None:
    """Format a metrics block."""
    print(f'\n{label}')
    print('-' * len(label))
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f'Fit time : {fit_seconds:.2f}s')


def build_master_node(comms: MockComms_master, logger: NullLogger, config: SimulationConfig, x_bounds: Tuple[float, float]) -> MasterNode:
    """Create and configure the unchanged MMLL MasterNode for DSVM."""
    master_node = MasterNode(pom=1, comms=comms, logger=logger, verbose=True)
    master_node.create_model_Master(
        'DSVM',
        model_parameters={
            'NC': config.nc,
            'Nmaxiter': config.nmaxiter,
            'tolerance': config.tolerance,
            'sigma': config.sigma,
            'C': config.c_value,
            'eps': config.eps,
            'NI': int(config_input_features(x_bounds, master_node)),
            'minvalue': float(x_bounds[0]),
            'maxvalue': float(x_bounds[1]),
        },
    )
    return master_node


def config_input_features(x_bounds: Tuple[float, float], master_node: MasterNode) -> int:
    """Placeholder helper to keep the create_model_Master call readable."""
    del x_bounds
    return master_node.NI


def start_worker_threads(
    partitions: Sequence[Tuple[np.ndarray, np.ndarray]],
    bus: ThreadedMessageBus,
    logger: NullLogger,
) -> Tuple[List[WorkerNode], List[threading.Thread], queue.Queue[Tuple[str, BaseException]]]:
    """Create workers, attach local partitions, and start their execution threads."""
    worker_nodes: List[WorkerNode] = []
    worker_threads: List[threading.Thread] = []
    error_queue: queue.Queue[Tuple[str, BaseException]] = queue.Queue()

    for worker_idx, (x_part, y_part) in enumerate(partitions):
        worker_id = f'worker_{worker_idx + 1}'
        worker_comms = MockComms_worker(bus, worker_id, logger)
        worker_node = WorkerNode(pom=1, comms=worker_comms, logger=logger, verbose=True)
        worker_node.set_training_data('mnist_binary', Xtr=x_part, ytr=y_part)
        worker_node.create_model_worker('DSVM')

        def worker_target(node: WorkerNode = worker_node, current_worker_id: str = worker_id) -> None:
            try:
                node.run()
            except BaseException as exc:
                error_queue.put((current_worker_id, exc))

        thread = threading.Thread(target=worker_target, name=worker_id)
        thread.start()

        worker_nodes.append(worker_node)
        worker_threads.append(thread)

    return worker_nodes, worker_threads, error_queue


def run_threaded_simulation(
    train_partitions: Sequence[Tuple[np.ndarray, np.ndarray]],
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: SimulationConfig,
) -> SimulationResult:
    """Run an unchanged MMLL DSVM training session over queue-backed comms."""
    worker_ids = [f'worker_{index + 1}' for index in range(len(train_partitions))]
    bus = ThreadedMessageBus(worker_ids)
    logger = NullLogger()

    _, worker_threads, error_queue = start_worker_threads(train_partitions, bus, logger)

    all_train_x = np.vstack([part[0] for part in train_partitions])
    master_comms = MockComms_master(bus, worker_ids, logger)
    master_node = MasterNode(pom=1, comms=master_comms, logger=logger, verbose=True, NI=all_train_x.shape[1])
    master_node.create_model_Master(
        'DSVM',
        model_parameters={
            'NC': config.nc,
            'Nmaxiter': config.nmaxiter,
            'tolerance': config.tolerance,
            'sigma': config.sigma,
            'C': config.c_value,
            'eps': config.eps,
            'NI': all_train_x.shape[1],
            'minvalue': float(all_train_x.min()),
            'maxvalue': float(all_train_x.max()),
        },
    )

    fit_error: List[BaseException] = []

    def master_target() -> None:
        try:
            master_node.fit(Xval=X_val, yval=y_val)
        except BaseException as exc:
            fit_error.append(exc)

    fit_start = time.perf_counter()
    master_thread = threading.Thread(target=master_target, name='master')
    master_thread.start()
    master_thread.join()
    fit_seconds = time.perf_counter() - fit_start

    if fit_error:
        master_node.MasterMLmodel.terminate_workers_()
        for thread in worker_threads:
            thread.join(timeout=2.0)
        raise fit_error[0]

    if not error_queue.empty():
        master_node.MasterMLmodel.terminate_workers_()
        worker_id, exc = error_queue.get()
        for thread in worker_threads:
            thread.join(timeout=2.0)
        raise RuntimeError(f'Worker thread failed: {worker_id}') from exc

    master_node.MasterMLmodel.terminate_workers_()
    for thread in worker_threads:
        thread.join(timeout=5.0)

    model = master_node.get_model()
    if model is None:
        raise RuntimeError('Training finished but MasterNode returned no model.')
    return SimulationResult(model=model, fit_seconds=fit_seconds, worker_threads=worker_threads)


def main() -> int:
    config = SimulationConfig()
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist_binary(config.random_state)

    federated_partitions = split_iid_among_workers(X_train, y_train, config.n_workers, config.random_state)
    print(f'\nStarting threaded federated DSVM with {config.n_workers} workers...')
    federated_result = run_threaded_simulation(federated_partitions, X_val, y_val, config)
    federated_metrics = compute_metrics(federated_result.model, X_test, y_test)
    print_metrics('Federated DSVM (3 workers)', federated_metrics, federated_result.fit_seconds)

    print('\nStarting centralized-equivalent DSVM baseline (single worker over full data)...')
    centralized_result = run_threaded_simulation([(X_train, y_train)], X_val, y_val, config)
    centralized_metrics = compute_metrics(centralized_result.model, X_test, y_test)
    print_metrics('Centralized-equivalent DSVM (1 worker)', centralized_metrics, centralized_result.fit_seconds)

    accuracy_gap = centralized_metrics['accuracy'] - federated_metrics['accuracy']
    print('\nComparison')
    print('----------')
    print(f'Federated accuracy   : {federated_metrics["accuracy"]:.4f}')
    print(f'Centralized accuracy : {centralized_metrics["accuracy"]:.4f}')
    print(f'Accuracy gap         : {accuracy_gap:+.4f}')

    if federated_metrics['accuracy'] < 0.95:
        print('\nWARNING: Federated accuracy is below the 95% target.')
        return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())