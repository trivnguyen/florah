
import logging
import torch
import pytorch_lightning as pl

from .linear_model import LinearModel

logger = logging.getLogger(__name__)

class DataModule(pl.LightningModule):

    def __init__(
        self, model_hparams, transform_hparams, optimizer_hparams,
        num_posteriors=5000):
        super().__init__()
        self.save_hyperparameters()
        logger.info("Run Hyperparameters:")
        for hparams in self.hparams:
            logger.info(f"{hparams}: {self.hparams[hparams]}")
        self.model = LinearModel(**model_hparams)
        self.transforms = None
        self.num_posteriors = num_posteriors

    def forward(self, x, *args, **kargs):
        return self.model(x, *args, **kargs)

    def configure_optimizers(self):
        """ Initialize optimizer and LR scheduler """
        optimizer = torch.optim.AdamW(
            self.parameters(), **self.hparams.optimizer_hparams)
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
        x, y = batch
        x =  self.model(x)
        log_prob = self.model.maf.log_prob(y, context=x)
        loss = -log_prob.mean()
        self.log('train_loss', loss, on_epoch=True, batch_size=len(x))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x =  self.model(x)
        log_prob = self.model.maf.log_prob(y, context=x)
        loss = -log_prob.mean()
        self.log('val_loss', loss, on_epoch=True, batch_size=len(x))
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        x =  self.model(x)
        y_pred = self.model.maf.sample(num_samples=self.num_posteriors, context=x)
        if y is not None:
            return y_pred, y
        return y_pred

