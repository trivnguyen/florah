
import torch

from . import flows

class DataModule(modules.BaseModule):
    """
    DataModule for Recurrent-MAF model
    """
    def __init__(
            self, model_hparams: Optional[dict] = None,
            transform_hparams: Optional[dict] = None,
            optimizer_hparams: Optional[dict] = None
        ) -> None:
        super(DataModule, self).__init__(
            MAF, Preprocess, model_hparams,
            transform_hparams, optimizer_hparams)

    def training_step(self, batch, batch_idx) -> FloatTensor:
        x, y = batch
        log_prob = self.model.log_prob(y, context=x)
        loss = -log_prob.mean()
        self.log('train_loss', loss, on_epoch=True, batch_size=len(x))
        return loss

    def validation_step(self, batch, batch_idx) -> FloatTensor:
        x, y = batch
        log_prob = self.model.log_prob(y, context=x)
        loss = -log_prob.mean()
        self.log('val_loss', loss, on_epoch=True, batch_size=len(x))
        return loss


class Preprocess(torch.nn.Module):
    """ Preprocessing module for Recurrent-MAF """
    def __init__(self, nx: int, ny: int) -> None:
        super(Preprocess, self).__init__()
        self.x_scaler = transforms.StandardScaler(nx, dim=0)
        self.y_scaler = transforms.StandardScaler(ny, dim=0)

    def forward(self, *args, **kargs):
        return self.transform(*args, **kargs)

    def transform(self, data, fit=True):
        """ Transform data """
        x = data["x"]
        y = data["y"]

        if isinstance(x, (list, tuple)):
            x = np.array(x)
        if isinstance(y, (list, tuple)):
            y = np.array(y)

        # reshape x
        x = x.reshape(-1, 1)

        # convert to tensor
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        # apply scaler as the last step
        if fit:
            self.x_scaler.fit(x)
            self.y_scaler.fit(y)
        x = self.x_scaler(x)
        y = self.y_scaler(y)

        return x, y

class MAF(torch.nn.Module):

    def __init__(
            self, features: int, hidden_features: int, context_features: int,
            num_layers: int, num_blocks: int, softplus: bool = True
        ) -> None:

        super(MAF, self).__init__()

        self.maf_blocks = flows.build_maf(
            features, hidden_features, context_features,
            num_layers, num_blocks)

    def forward(self, x):
        return x

    def log_prob(
            self, x: FloatTensor, context: FloatTensor) -> FloatTensor:
        """ Return MAF log-likelihood P(x | context)"""
        return self.maf_blocks.log_prob(x, context=context)
