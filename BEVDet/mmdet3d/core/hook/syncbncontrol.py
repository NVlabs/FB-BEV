# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook
from mmdet3d.core.hook.utils import is_parallel
from torch.nn import SyncBatchNorm

__all__ = ['SyncbnControlHook']


@HOOKS.register_module()
class SyncbnControlHook(Hook):
    """ """

    def __init__(self, syncbn_start_epoch=1):
        super().__init__()
        self.is_syncbn=False
        self.syncbn_start_epoch = syncbn_start_epoch

    def cvt_syncbn(self, runner):
        if is_parallel(runner.model.module):
            runner.model.module.module=\
                SyncBatchNorm.convert_sync_batchnorm(runner.model.module.module,
                                                     process_group=None)
        else:
            runner.model.module=\
                SyncBatchNorm.convert_sync_batchnorm(runner.model.module,
                                                     process_group=None)

    def before_train_epoch(self, runner):
        if runner.epoch>= self.syncbn_start_epoch and not self.is_syncbn:
            print('start use syncbn')
            self.cvt_syncbn(runner)
            self.is_syncbn=True

