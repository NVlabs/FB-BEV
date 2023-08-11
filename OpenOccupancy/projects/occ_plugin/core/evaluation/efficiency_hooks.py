import copy
from mmcv.runner import HOOKS, Hook
import time
try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')
import torch
import torch.distributed as dist

@HOOKS.register_module()
class OccEfficiencyHook(Hook):
    def __init__(self, dataloader,  **kwargs):
        self.dataloader = dataloader 
        self.warm_up = 5
        
    def construct_input(self, DUMMY_SHAPE=None, m_info=None):
        if m_info is None:
            m_info = next(iter(self.dataloader))
        img_metas = m_info['img_metas'].data
        input = dict(
            img_metas=img_metas,
        )
        if 'img_inputs' in m_info.keys():
            img_inputs = m_info['img_inputs']
            for i in range(len(img_inputs)):
                if isinstance(img_inputs[i], list):
                    for j in range(len(img_inputs[i])):
                        img_inputs[i][j] = img_inputs[i][j].cuda()
                else:
                    img_inputs[i] = img_inputs[i].cuda()
            input['img_inputs'] = img_inputs
            
        if 'points' in m_info.keys():
            points = m_info['points'].data[0]
            points[0] = points[0].cuda()
            input['points'] = points
        return input
    
    def before_run(self, runner):
        torch.cuda.reset_peak_memory_stats()
        model = copy.deepcopy(runner.model)
        if hasattr(model, 'module'):
            model = model.module
        if hasattr(model, 'forward_dummy'):
            model.forward_train = model.forward_dummy
            model.forward_test = model.forward_dummy
            model.eval()
        else:
            raise NotImplementedError(
                'FLOPs counter is currently not supported for {}'.format(
                    model.__class__.__name__))
        # inf time
        pure_inf_time = 0
        itv_sample = 10
        for i, data in enumerate(self.dataloader):
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                model(return_loss=False, rescale=True, **self.construct_input(m_info=data))

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            if i >= self.warm_up:
                pure_inf_time += elapsed
                if (i + 1) % itv_sample == 0:
                    fps = (i + 1 - self.warm_up) / pure_inf_time
                    if runner.rank == 0:
                        runner.logger.info(f'Done sample [{i + 1:<3}/ {itv_sample*5}], '
                            f'fps: {fps:.1f} sample / s')

            if (i + 1) == itv_sample*5:
                pure_inf_time += elapsed
                fps = (i + 1 - self.warm_up) / pure_inf_time
                if runner.rank == 0:
                    runner.logger.info(f'Overall fps: {fps:.1f} sample / s')
                break
            
        # flops and params   
        if runner.rank == 0:
            flops, params = get_model_complexity_info(
                model, (None, None), input_constructor=self.construct_input)
            
            split_line = '=' * 30
            gpu_measure = torch.cuda.max_memory_allocated() / 1024. / 1024. /1024. 
            runner.logger.info(f'{split_line}\n' f'Flops: {flops}\nParams: {params}\nGPU memory: {gpu_measure:.2f}GB\n{split_line}')
            
        if dist.is_available() and dist.is_initialized():
            dist.barrier() 
        
        
    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass
  
