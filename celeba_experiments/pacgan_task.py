from gpu_task_scheduler.gpu_task import GPUTask

class PacGANTask(GPUTask):

    def main(self):
        from pacgan_task_wrapper_celebA import PacGANTaskWrapper
        wrapper = PacGANTaskWrapper(self._work_dir, self._config)
        wrapper.main()