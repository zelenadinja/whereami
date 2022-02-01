#  type: ignore
import os
from typing import Dict, List

import torch
import wandb
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import pandas as pd
import numpy as np

from dataset.dataset import LandmarkDataset
from dataset.augmentations import aug_version_1
from models.networks import LandmarkResidual
from models.utils import save_checkpoint_to_s3
from models.losses import criterion
from src.utils import read_artifacts_s3, set_seed, artifact_to_s3
from src.train import train_epoch, validate_epoch


def main(run_name) -> None:

    with wandb.init(project="landmarkrecognition", name=run_name):

        load_dotenv()
        args = read_artifacts_s3(object_key=os.environ.get("CONFIG_VERSION_3"))
        set_seed(args["seed"])
        df = pd.read_csv(args["df_path"])
        train, valid = train_test_split(
            df, test_size=args["valid_split"],
            shuffle=True, random_state=args["seed"]
        )

        train_dataset = LandmarkDataset(
            dataframe=train, transform=aug_version_1(args, train=True)
            )
        valid_dataset = LandmarkDataset(
            dataframe=valid, transform=aug_version_1(args, train=True)
            )

        class_sample_count = np.unique(
            train_dataset.targets, return_counts=True
        )[1]
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in train_dataset.targets])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=samples_weight, num_samples=len(samples_weight)
        )

        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args["train_batch"],
            sampler=sampler,
            num_workers=args['workers'],
            pin_memory=True,
        )
        validloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args["valid_batch"],
            shuffle=True,
            num_workers=args['workers'],
            pin_memory=True,
        )
        device = torch.device(args["device"])
        model = LandmarkResidual(
            model=args["model"],
            pretrained=args["pretrained"],
            num_classes=args["num_classes"],
        )
        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args["lr"], weight_decay=args['decay']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args['factor'],
            patience= args['patience'], verbose=True,
        )
        loss_fn = criterion()

        best_loss = np.inf
        best_f1 = 0
        best_acc = 0

        history: Dict[str, List[float]] = dict()
        history['training_acc'] = list()
        history['training_losses'] = list()
        history['training_f1scores'] = list()
        history['validation_acc'] = list()
        history['validation_losses'] = list()
        history['validation_f1scores'] = list()

        for epoch in range(args["num_epochs"]):

            train_loss, train_acc, train_f1 = train_epoch(
                model,
                trainloader,
                optimizer,
                loss_fn,
                args['grad_accum'],
                device,
                args["num_classes"],
                epoch + 1,
                args["log_freq"],
            )
            valid_loss, valid_acc, valid_f1 = validate_epoch(
                model,
                validloader,
                loss_fn,
                device,
                args["num_classes"],
                epoch + 1,
                args["log_freq"],
            )
            scheduler.step(valid_loss)
            
            wandb.log(
                {
                    "train_epochs_losses": train_loss,
                    "train_epochs_accuracies": train_acc,
                    "train_epochs_f1score": train_f1,
                    "valid_epochs_losses": valid_loss,
                    "valid_epochs_accuracies": valid_acc,
                    "valid_epochs_f1score": valid_f1,
                }
            )
            print(
                f"Epoch:{epoch+1} Train Loss:{train_loss:.4f} Valid Loss:{valid_loss:.4f}\
                Train Acc:{train_acc:.4f} Valid Acc:{valid_acc:.4f}\
                Train F1:{train_f1:.4f} Valid F1:{valid_f1:.4f}"
            )

            history['training_acc'].append(float(train_acc))
            history['training_losses'].append(float(train_loss))
            history['training_f1scores'].append(float(train_f1))
            history['validation_acc'].append(float(valid_acc))
            history['validation_losses'].append(float(valid_loss))
            history['validation_f1scores'].append(float(valid_f1))

            if valid_acc > best_acc:
                wandb.summary['best_validation_accuracy'] = valid_loss
                best_acc = valid_acc

            if valid_f1 > best_f1:
                wandb.summary['best_validation_f1score'] = valid_f1
                best_f1 = valid_f1

            if valid_loss < best_loss:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': valid_loss,
                    'acc': best_acc,
                    'f1score': best_f1,
                }
                save_checkpoint_to_s3(
                    checkpoint=checkpoint,
                    checkpoint_name=f"{run_name}_e{epoch+1}"
                )
                print(
                    f'Validation loss decreased from \
                    {best_loss:4f} to {valid_loss:.4f}'
                )
                wandb.summary['best_validation_loss'] = valid_loss
                best_loss = valid_loss

        artifact_to_s3(
            history, bucket=os.environ.get('S3_BUCKET'),
            key=f'run_artifacts/{run_name}'
            )


if __name__ == '__main__':
    main(run_name='VERSION_3')
