import pytorch_lightning as pl
from timm.optim import AdamW, Nadam
from timm.optim.madgrad import MADGRAD
import torch
from torch import nn


from src.nn.eval import SmapeLoss


class LitModel(pl.LightningModule):
    """PL Model"""
    def __init__(
        self,
        config,
        base_model,
    ):
        super().__init__()
        self.config = config
        self.model = base_model
        self.criterion = torch.nn.SmoothL1Loss(beta=self.config.loss.beta)
        self.metric = SmapeLoss()

    def forward(self, x, pop):
        return self.model(x, pop)

    def step(self, batch):
        x, y, pop, norm = batch
        yhat = self.forward(x, pop)
        loss = self.criterion(yhat, y)
        return loss, yhat, y, pop, norm

    def training_step(self, batch, batch_idx):
        loss, preds, y, _, _ = self.step(batch)
        loss_corr = self.criterion(y*self.config.loss.loss_mult, preds*self.config.loss.loss_mult)
        self.log("train/loss", loss_corr, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, preds, y, pop, norm = self.step(batch)
        loss_corr = self.metric( (y + 1) * norm * 100 / pop, (preds + 1) * norm * 100 / pop)

        self.log("val/loss", loss_corr, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def predict_step(self, batch, batch_idx):
        loss, preds, y, pop, norm = self.step(batch)
        return (preds + 1) * norm * 100 / pop

    def configure_optimizers(self):
        if self.config.opti.name == 'madgrad':
            optimizer = MADGRAD(self.parameters(), lr=self.config.opti.lr, weight_decay=self.config.opti.wd)
        else:
            optimizer = Nadam(self.parameters(), lr=self.config.opti.lr, weight_decay=self.config.opti.wd)

        if self.config.scheduler.name == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                            total_steps=self.trainer.estimated_stepping_batches,
                                                            max_lr=self.config.opti.lr,
                                                            div_factor=self.config.scheduler.get('div_factor', 10),
                                                            pct_start=self.config.scheduler.get('pct_start', 0),
                                                            final_div_factor=self.config.scheduler.get('final_div_factor', 100))
            
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=self.config.scheduler.gamma,
                                                         milestones=self.config.scheduler.steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
