def main():
    """Real full-training throughput probe; suppress only checkpoint disk writes."""
    import json,os,pathlib,sys
    import torch
    from flux_action.training.trainer import TrainConfig,Trainer
    
    class ProfileTrainer(Trainer):
        def _checkpoint(self):
            self._log_line({'event':'profile_checkpoint_skipped','step':self.step})
    
    cfg=TrainConfig.from_file(sys.argv[1])
    trainer=ProfileTrainer(cfg)
    print(json.dumps({'event':'PROFILE_RANK_READY','rank':trainer.rank,'world_size':trainer.world_size,'device':str(trainer.device),'gpu':torch.cuda.get_device_name()}),flush=True)
    trainer.run()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
