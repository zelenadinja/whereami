import os

import torch
import wandb
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

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
        args = read_artifacts_s3(object_key=os.environ.get("VERSION_0"))
        set_seed(args["seed"])
        df = pd.read_csv(args["df_path"])
        train, valid = train_test_split(
            df, test_size=args["valid_split"], shuffle=True, random_state=args["seed"]
        )
        train_dataset = LandmarkDataset(dataframe=train, transform=aug_version_1(args))
        valid_dataset = LandmarkDataset(dataframe=valid, transform=aug_version_1(args))
        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args["train_batch"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        valiloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args["valid_batch"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        device = torch.device(args["device"])
        model = LandmarkResidual(
            model=args["model"],
            pretrained=args["pretrained"],
            num_classes=args["num_classes"],
        )
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
        loss_fc = criterion()

        best_loss = np.inf

        history = dict()
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
                loss_fc,
                device,
                args["num_classes"],
                epoch + 1,
                args["log_freq"],
            )
            valid_loss, valid_acc, valid_f1 = validate_epoch(
                model,
                validloader,
                criterion,
                device,
                args["num_classes"],
                epoch + 1,
                args["log_freq"],
            )

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
                f"Epoch:{epoch+1} Train Loss:{train_loss:.f} Valid Loss:{valid_loss:.4f} Train Acc:{train_acc:.4f} Valid Acc:{valid_acc:4f} Train F1:{train_f1:.4f} Valid F1:{valid_f1:.4f} "
            )

            history['training_acc'].append(train_acc)
            history['training_losses'].append(train_loss)
            history['training_f1scores'].append(train_f1)
            history['validation_acc'].append(valid_acc)
            history['validation_losses'].append(valid_loss)
            history['validation_f1scores'].append(valid_f1)


            if valid_loss < best_loss:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss':valid_loss,
                    'acc':valid_acc,
                    'f1score':valid_f1,
                }
                save_checkpoint_to_s3(checkpoint=checkpoint, checkpoint_name=f"{run_name}_e{epoch+1}")
                print(f'Validation loss decreased from {best_loss:4f} to {valid_loss:.4f}')
                wandb.summary['best_validation_loss'] = valid_loss
                best_loss = valid_loss

        artifact_to_s3(history, bucket=os.environ.get('S3_BUCKET'), key=f'run_artifacts/{run_name}')

if __name__ == '__main__':
    main(run_name='VERSION_1')
    




