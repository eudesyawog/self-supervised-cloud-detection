import os
import torch
import torchvision

import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from datetime import datetime
from argparse import ArgumentParser
from typing import Union, Any
from pl_bolts.metrics import mean, precision_at_k
from pl_bolts.models.self_supervised import Moco_v2
from ..losses.functional import infoNCE
from ..tools.utils import modify_choices
from pytorch_lightning.strategies import DDPStrategy

class CustomDepthMoco_v2(Moco_v2):
    def init_encoders(self, base_encoder):
        template_model = getattr(torchvision.models, base_encoder)
        encoder_q = template_model(num_classes=self.hparams.emb_dim)
        encoder_k = template_model(num_classes=self.hparams.emb_dim)

        encoder_q.conv1 = nn.Conv2d(self.hparams.input_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        encoder_k.conv1 = nn.Conv2d(self.hparams.input_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)

        nn.init.kaiming_normal_(encoder_q.conv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(encoder_k.conv1.weight, mode="fan_out", nonlinearity="relu")

        return encoder_q, encoder_k

    def forward(self, img_q, img_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(img_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            # if self._use_ddp_or_ddp2(self.trainer):
            if isinstance(self.trainer.strategy, DDPStrategy):
                img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

            k = self.encoder_k(img_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            # if self._use_ddp_or_ddp2(self.trainer):
            if isinstance(self.trainer.strategy, DDPStrategy):
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        return q, k

class MoCo(pl.LightningModule):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.moco = CustomDepthMoco_v2(*args, **kwargs)

    def forward(self, img_q, img_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            encoded q and k
        """

        q, k = self.moco(img_q, img_k)

        return q, k

    def training_step(self, batch, batch_idx):
        (img_q, img_k), _ = batch

        # Update the momentum encoders
        self.moco._momentum_update_key_encoder()

        qs, ks = self(img_q=img_q, img_k=img_k)

        # Compute the standard single modality MoCo losses
        loss, target, output = infoNCE(qs, ks, self.moco.queue, self.hparams.softmax_temperature, True)

        self.moco._dequeue_and_enqueue(ks, queue=self.moco.queue, queue_ptr=self.moco.queue_ptr)

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        log = {
            "train_loss": loss,
            "train_acc_1": acc1,
            "train_acc5": acc5
        }

        self.log_dict(log,sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (img_q, img_k), _ = batch

        qs, ks = self(img_q=img_q, img_k=img_k)

        # Compute the standard single modality MoCo losses
        loss, target, output = infoNCE(
            qs, ks, self.moco.val_queue, self.hparams.softmax_temperature, True
        )
        
        self.moco._dequeue_and_enqueue(ks, queue=self.moco.val_queue, queue_ptr=self.moco.val_queue_ptr)
        
        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        log = {
            "val_loss": loss,
            "val_acc1": acc1,
            "val_acc5": acc5
        }

        return log

    def training_epoch_end(self, outputs) -> None:
        return super().training_epoch_end(outputs)

    def validation_epoch_end(self, outputs):
        log = {}

        log["val_loss"] = mean(outputs, "val_loss")

        log["val_acc1"] = mean(outputs, "val_acc1")
        log["val_acc5"] = mean(outputs, "val_acc5")

        self.log_dict(log,sync_dist=True)

    def configure_optimizers(self):
        # This is dirty and not recommended, but it is safe only in this case where MoCo just needs to pull
        # attributes from the Trainer and never sets state.
        if not self.moco.trainer:
            self.moco.trainer = self.trainer

        optimizer = torch.optim.SGD(
            self.parameters(),
            self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.trainer.max_epochs,
        )

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = Moco_v2.add_model_specific_args(parent_parser)
        parser.add_argument("--input_ch", type=int, default=13)
        parser.add_argument("--patch_size", type=int, default=256)
        modify_choices(parser, "dataset", ["SEN12MS"])
        return parser