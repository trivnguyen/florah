
import logging
import torch
import pytorch_lightning as pl

logger = logging.getLogger(__name__)

class DataModule(pl.LightningModule):

    def __init__(
        self, model_hparams, optimizer_hparams):
        super().__init__()
        self.save_hyperparameters()
        logger.info("Run Hyperparameters:")
        for hparams in self.hparams:
            logger.info(f"{hparams}: {self.hparams[hparams]}")
        self.model = None
        self.loss_fn = None

    def forward(self, x, *args, **kargs):
        return self.model(x, *args, **kargs)

    def configure_optimizers(self):
        """ Initialize optimizer and LR scheduler """
        optimizer = torch.optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
               'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 'min', patience=4),
                'monitor': 'train_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def training_step(self, batch, batch_idx):
        x = self.model(batch.x)
        loss = self.loss_fn(x, batch.y)
        self.log('train_loss', loss, on_epoch=True, batch_size=len(batch.x))
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.model(batch.x)
        loss = self.loss_fn(x, batch.y)
        self.log('val_loss', loss, on_epoch=True, batch_size=len(batch.x))
        return loss

