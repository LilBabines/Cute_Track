from lightning.pytorch.callbacks import Callback


class Record_train_dynamics(Callback):
    """
    Callback to record training dynamics (losses) for each instance during training (after each train_step).
    Stores the results in a dictionary: {instance_id: []} 
    """

    def __init__(self):
        super().__init__()
        self.train_dynamics = {}  # {instance_id: [loss_epoch1, loss_epoch2, ...]}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Called when the train batch ends.
        Args:
            trainer: the Trainer
            pl_module: the LightningModule
            outputs: the outputs of the training step
            batch: the batch data
            batch_idx: index of the batch
            dataloader_idx: index of the dataloader
        """
        # Assuming batch contains 'instance_ids', 'logits', and 'losses'
        instance_ids = batch['idx']  # shape (B,)
        losses = outputs['all_losses']              # shape (B,)

        for i, instance_id in enumerate(instance_ids):
            instance_id = instance_id.item()
            loss = losses[i].item()

            if instance_id not in self.train_dynamics:
                self.train_dynamics[instance_id] = []

            self.train_dynamics[instance_id].append(loss)
            pl_module.log(f"train_dynamics/loss_instance_{instance_id}", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

