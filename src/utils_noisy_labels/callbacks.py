from lightning.pytorch.callbacks import Callback


class Record_train_dynamics(Callback):
    """
    Callback to record training dynamics (losses and logits) for each instance during training (after each train_step).
    Stores the results in a dictionary: {instance_id: {'logits': [], 'losses': []}} 
    """

    def __init__(self):
        super().__init__()
        self.train_dynamics = {}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
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
        instance_ids = batch['instance_ids']  # shape (B,)
        logits = outputs['logits']            # shape (B, ...)
        losses = outputs['loss']              # shape (B,)

        for i, instance_id in enumerate(instance_ids):
            instance_id = instance_id.item()
            logit = logits[i].detach().cpu().numpy()
            loss = losses[i].item()

            if instance_id not in self.train_dynamics:
                self.train_dynamics[instance_id] = {'logits': [], 'losses': []}

            self.train_dynamics[instance_id]['logits'].append(logit)
            self.train_dynamics[instance_id]['losses'].append(loss)