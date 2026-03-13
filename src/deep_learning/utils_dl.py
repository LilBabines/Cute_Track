from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger


class Record_train_dynamics(Callback):
    """
    Callback to record training dynamics (losses) for each instance
    during training and validation.
    """

    def __init__(self, save_dir="logs/", name="train_dynamics"):
        super().__init__()
        self.train_dynamics = {}
        self.val_dynamics = {}

        self.csv_logger_train = CSVLogger(
            save_dir=save_dir, name=name, version="train_dynamics"
        )
        self.csv_logger_val = CSVLogger(
            save_dir=save_dir, name=name, version="val_dynamics"
        )

        self.row_train = {}
        self.row_val = {}

    # ── Train ──────────────────────────────────────────────

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        
    

        instance_ids = batch['idx']
        img_name = batch['img_name']

        losses = outputs['all_losses']

        for i, instance_id in enumerate(instance_ids):
            instance_id = instance_id.item()
            img_name_i = img_name[i]
            loss = losses[i].item()

            if instance_id not in self.train_dynamics:
                self.train_dynamics[instance_id] = []
            self.train_dynamics[instance_id].append(loss)

            self.row_train[f"{img_name_i}"] = round(loss, 3)

    def on_train_epoch_end(self, trainer, pl_module):
        self.csv_logger_train.log_metrics(self.row_train, step=trainer.current_epoch)
        self.csv_logger_train.save()
        self.row_train = {}

    # ── Validation ─────────────────────────────────────────

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        
        instance_ids = batch['idx']
        img_name = batch['img_name']
        losses = outputs['all_losses']

        for i, instance_id in enumerate(instance_ids):
            instance_id = instance_id.item()
            img_name_i = img_name[i]
            loss = losses[i].item()

            if instance_id not in self.val_dynamics:
                self.val_dynamics[instance_id] = []
            self.val_dynamics[instance_id].append(loss)

            self.row_val[f"{img_name_i}"] = round(loss, 3)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.csv_logger_val.log_metrics(self.row_val, step=trainer.current_epoch)
        self.csv_logger_val.save()
        self.row_val = {}

    # ── Fin ────────────────────────────────────────────────

    def on_train_end(self, trainer, pl_module):
        self.csv_logger_train.experiment.save()
        self.csv_logger_val.experiment.save()