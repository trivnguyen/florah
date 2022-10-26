
import logging
logger = logging.getLogger(__name__)

import torch
import pytorch_lightning as pl

from .model import MergerTreeRNN
from .transform import Transform

class DataModule(pl.LightningModule):

    def __init__(self, model_hparams, transform_hparams, optimizer_hparams,
                 num_posteriors=5000):
        super().__init__()
        self.save_hyperparameters()
        logger.info("Run Hyperparameters:")
        for hparams in self.hparams:
            logger.info(f"{hparams}: {self.hparams[hparams]}")
        self.transform = Transform(**transform_hparams)
        self.model = MergerTreeRNN(**model_hparams)
        self.num_posteriors = num_posteriors

    def forward(self, x, *args, **kargs):
        return self.model(x, *args, **kargs)

    def configure_optimizers(self):
        """ Initialize optimizer and LR scheduler """
        optimizer = torch.optim.AdamW(
            self.parameters(), **self.hparams.optimizer_hparams['optimizer'])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 'min',
                    patience=self.hparams.optimizer_hparams['scheduler']['patience']),
                'monitor': 'train_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def training_step(self, batch, batch_idx):
        x, y, seq_len, mask = batch
        batch_dim, max_len, _ = x.shape

        # get context
        context, _ = self(x, seq_len)
        context = context.reshape(batch_dim * max_len, -1)

        # get mask
        y = y.reshape(batch_dim * max_len, -1)
        mask = mask.reshape(-1)

        # calculate log likelihood and loss function
        log_prob = self.model.maf.log_prob(y[mask], context=context[mask])
        loss = -log_prob.mean()

        self.log('train_loss', loss, on_epoch=True, batch_size=len(x))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, seq_len, mask = batch
        batch_dim, max_len, _ = x.shape

        # get context
        context, _ = self(x, seq_len)
        context = context.reshape(batch_dim * max_len, -1)

        # get mask
        y = y.reshape(batch_dim * max_len, -1)
        mask = mask.reshape(-1)

        # calculate log likelihood and loss function
        log_prob = self.model.maf.log_prob(y[mask], context=context[mask])
        loss = -log_prob.mean()

        self.log('val_loss', loss, on_epoch=True, batch_size=len(x))
        return loss

