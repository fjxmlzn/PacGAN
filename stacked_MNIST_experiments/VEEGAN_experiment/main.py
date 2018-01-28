from pacgan_task import PacGANTask

if __name__ == "__main__":
    from config import config
    from gpu_task_scheduler.gpu_task_scheduler import GPUTaskScheduler
    
    scheduler = GPUTaskScheduler(config=config, gpu_task_class=PacGANTask)
    scheduler.start()