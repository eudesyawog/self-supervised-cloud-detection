import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from argparse import ArgumentParser
from pl_bolts.metrics import mean
from pl_bolts.models.self_supervised import resnets
from torchmetrics.classification import BinaryJaccardIndex, BinaryAccuracy, MulticlassJaccardIndex, MulticlassAccuracy

from ..losses.functional import soft_dice_loss_with_logits, multi_soft_dice_loss_with_logits

def set_stride_recursive(module, stride):
    if hasattr(module, "stride"):
        module.stride = (stride, stride)

    for child in module.children():
        set_stride_recursive(child, stride)


class ResNet(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 1,
        backbone: str = "resnet18",
        input_ch: int = 13,
        segmentation: bool = True,
        classification_head: nn.Module = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["classification_head"])

        self.num_classes = num_classes
        self.segmentation = segmentation

        # Create the base model
        template_model = getattr(resnets, backbone)
        self.model = template_model(num_classes=num_classes, return_all_feature_maps=self.segmentation)

        self.model.conv1 = nn.Conv2d(input_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Initialize this correctly
        nn.init.kaiming_normal_(self.model.conv1.weight, mode="fan_out", nonlinearity="relu")

        self.model.fc = nn.Identity()

        if self.segmentation:
            set_stride_recursive(self, 1)

        self.classifier = classification_head
        self.iou_metric_train = BinaryJaccardIndex() if self.num_classes == 1 else MulticlassJaccardIndex(num_classes=self.num_classes)
        self.iou_metric_val = BinaryJaccardIndex() if self.num_classes == 1 else MulticlassJaccardIndex(num_classes=self.num_classes)
        self.oa_train = BinaryAccuracy() if self.num_classes == 1 else MulticlassAccuracy(num_classes=self.num_classes)
        self.oa_val = BinaryAccuracy() if self.num_classes == 1 else MulticlassAccuracy(num_classes=self.num_classes)

        self.dice_loss_fn = soft_dice_loss_with_logits if self.num_classes == 1 else multi_soft_dice_loss_with_logits 
        self.classif_loss = F.binary_cross_entropy_with_logits if self.num_classes == 1 else F.cross_entropy

    def load_from_checkpoint(
        self,
        checkpoint: str,
        filter_and_remap: str = None,
    ):
        ckpt = torch.load(checkpoint)

        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        if filter_and_remap:
            state_dict = {
                k.replace(filter_and_remap, "model"): v for k, v in state_dict.items() if filter_and_remap in k
            }

        return self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        # Get the last layer returned by the basemodel
        y_pred = self.model(x)[-1]

        if self.classifier:
            y_pred = self.classifier(y_pred)

        return y_pred

    def _common_step(
        self,
        batch,
        batch_idx,
        include_preds=False,
    ):
        x, y_true = batch
        y_pred = self(x)

        if self.num_classes > 1:
            y_true = y_true.long().squeeze() # Target needed to be int for cross entropy loss
    
        loss = self.dice_loss_fn(y_pred, y_true) + self.classif_loss(y_pred, y_true)

        return loss, y_pred, y_true

    def training_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._common_step(batch, batch_idx, False)
        iou = self.iou_metric_train(y_pred, y_true)
        oa = self.oa_train(y_pred, y_true)

        log = {
            "loss": loss,
            "acc": oa,
            "iou": iou,
        }

        return log

    def training_epoch_end(self, outputs):
        train_loss = mean(outputs, "loss")

        log = {
            "train_loss": train_loss,
            "train_acc": self.oa_train.compute(),
            "train_iou": self.iou_metric_train.compute(),
        }

        self.log_dict(log)

    def validation_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._common_step(batch, batch_idx, True)
        iou = self.iou_metric_val(y_pred, y_true)
        oa = self.oa_val(y_pred, y_true)

        log = {
            "loss": loss,
            "acc": oa,
            "iou": iou,
        }

        return log

    def validation_epoch_end(self, outputs):
        val_loss = mean(outputs, "loss")

        log = {
            "val_loss": val_loss,
            "val_acc": self.oa_val.compute(),
            "val_iou": self.iou_metric_val.compute(),
        }
        self.log_dict(log)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        loss, y_pred, y_true = self._common_step(batch, batch_idx, True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._common_step(batch, batch_idx, True)
        return {"loss": loss}

    def test_epoch_end(self, outputs):
        test_loss = mean(outputs, "loss")

        log = {"test_loss": test_loss}
        self.log_dict(log)

    def configure_optimizers(self):
        if "optimizer" in self.hparams and self.hparams.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                self.hparams.learning_rate,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()),
                self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        if "scheduler" in self.hparams:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                self.trainer.max_epochs,
            )
        else:
            scheduler = None

        if scheduler:
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--backbone", type=str, default="resnet18")
        parser.add_argument("--num_classes", type=int, default=1)
        parser.add_argument("--input_ch", type=int, default=13)
        parser.add_argument("--segmentation", action="store_true")
        parser.add_argument("--classification_head", type=str, default=None)
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-8)
        return parser
    
