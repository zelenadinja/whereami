import torch
import torchmetrics
import tqdm
import wandb

from src.utils import AverageMeter


def train_epoch(
        model, loader, optimizer, criterion,
        device, grad_accum, num_classes,
         epoch, log_freq,

):
    """Forward  and backward pass"""
    model.train()
    scaler = torch.cuda.amp.GradScaler()

    train_acc = torchmetrics.Accuracy().to(device)
    train_f1 = torchmetrics.F1Score(
        num_classes=num_classes, average="weighted"
    ).to(device)
    train_loss = AverageMeter()
    pbar = tqdm.tqdm(
        enumerate(loader),
        total=len(loader),
        unit="batches",
        position=0,
        leave=True,
        desc=f"Training epoch {epoch}",
    )

    for batch_idx, (images, labels) in pbar:

        images, labels = images.to(device), labels.to(device)

        model.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / grad_accum

        scaler.scale(loss).backward()
        if (batch_idx + 1) % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()

        _, predictions = torch.max(outputs.detach(), dim=1)
        train_loss.update(loss.detach(), images.size(0))
        batch_acc = train_acc(predictions, labels.detach())
        batch_f1 = train_f1(predictions, labels.detach())

        if (batch_idx + 1) % log_freq == 0:
            wandb.log(
                {
                    f"train_batch_loss_{epoch}": loss.detach(),
                    f"train_batch_accuracy_{epoch}": batch_acc,
                    f"train_batch_f1_score_{epoch}": batch_f1,
                }
            )

    return train_loss.avg, train_acc.compute(), train_f1.compute()


def validate_epoch(
        model, loader, criterion, device, num_classes, epoch, log_freq
):  # pylint: disable=too-many-arguments
    """forward pass"""

    model.eval()

    valid_acc = torchmetrics.Accuracy().to(device)
    valid_f1 = torchmetrics.F1Score(
        num_classes=num_classes, average="weighted"
    ).to(device)
    valid_loss = AverageMeter()
    pbar = tqdm.tqdm(
        enumerate(loader),
        total=len(loader),
        unit="batches",
        position=0,
        leave=True,
        desc=f"Validation epoch {epoch}",
    )

    with torch.no_grad():
        for batch_idx, (images, labels) in pbar:

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predictions = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)
            valid_loss.update(loss, images.size(0))
            batch_acc = valid_acc(predictions, labels)
            batch_f1 = valid_f1(predictions, labels)

            if (batch_idx + 1) % log_freq == 0:
                wandb.log(
                    {
                     f"valid_batch_loss_{epoch}": loss,
                     f"valid_batch_accuracy_{epoch}": batch_acc,
                     f"valid_batch_f1_score_{epoch}": batch_f1,
                    }
                )

    return valid_loss.avg, valid_acc.compute(), valid_f1.compute()