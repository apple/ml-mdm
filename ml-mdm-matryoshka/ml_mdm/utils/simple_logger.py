# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
import os

import matplotlib.pyplot as plt

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    from torch.utils.tensorboard.writer import SummaryWriter


class LoggerBase:
    def __init__(self, output_dir, logging_freq):
        self._batch_num = 0
        self._output_dir = output_dir
        self._logging_freq = logging_freq
        self.next_logger = None
        self.call_backs = []
        self._last_step_batch_num = {}

    @property
    def batch_num(self):
        return self._batch_num

    @batch_num.setter
    def batch_num(self, value):
        self._batch_num = value

    def add_figure(self, name, fig):
        raise NotImplementedError("Derived classes to implement")

    def add_scalar(self, name, value):
        raise NotImplementedError("Derived classes to implement")

    def add_scalars(self, name, value):
        raise NotImplementedError("Derived classes to implement")

    def add_callback(self, callback):
        self.call_backs.append(callback)


class Logger(LoggerBase):
    def __init__(self, output_dir, logging_freq):
        super(Logger, self).__init__(output_dir, logging_freq)

    def add_tensorboard_logger(self):
        tb_logger = TensorboardLogger(self._output_dir, self._logging_freq)
        tb_logger.batch_num = self.batch_num

        tb_logger.next_logger = self.next_logger
        self.next_logger = tb_logger

    @property
    def batch_num(self):
        return self._batch_num

    @batch_num.setter
    def batch_num(self, value):
        self._batch_num = value

        next_logger = self.next_logger
        while next_logger is not None:
            next_logger.batch_num = value
            next_logger = next_logger.next_logger

    def needs_update(self, name):
        if name in self._last_step_batch_num and self._batch_num < (
            self._last_step_batch_num[name] + self._logging_freq
        ):
            return False
        self._last_step_batch_num[name] = self._batch_num
        return True

    def add_scalar(self, name, value):
        if not self.needs_update(name):
            return
        next_logger = self.next_logger
        while next_logger is not None:
            next_logger.add_scalar(name, value)
            next_logger = next_logger.next_logger

    def add_figure(self, name, value):
        if not self.needs_update(name):
            return
        next_logger = self.next_logger
        while next_logger is not None:
            next_logger.add_figure(name, value)
            next_logger = next_logger.next_logger

    def add_scalars(self, name, value):
        if not self.needs_update(name):
            return
        next_logger = self.next_logger
        while next_logger is not None:
            next_logger.add_scalars(name, value)
            next_logger = next_logger.next_logger

    def execute_callbacks(self):
        for callback in self.call_backs:
            callback(self)


class TensorboardLogger(LoggerBase):
    def __init__(self, output_dir, logging_freq):
        super(TensorboardLogger, self).__init__(output_dir, logging_freq)
        self.tb_writer = SummaryWriter(log_dir=self._output_dir)

    def add_scalar(self, name, value):
        self.tb_writer.add_scalar(name, value, self.batch_num)

    def add_figure(self, name, value):
        self.tb_writer.add_figure(name, value, self.batch_num)

    def add_scalars(self, name, value):
        self.tb_writer.add_scalars(name, value, self.batch_num)
