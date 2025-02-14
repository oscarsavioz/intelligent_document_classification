from classes.load_models import MobileNetV3Model
from torchvision.transforms import transforms
import torch
import lightning.pytorch as pl
import argparse
import warnings
import sys
import os
from classes.datasets import RVLCDIPDataModule

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-p", "--pretrained", type=str, required=True)
    parser.add_argument("-b", "--batch_size", type=int, required=True)
    parser.add_argument("-e", "--epochs", type=int, required=False, default=10)
    parser.add_argument("-l", "--learning_rate", type=float, required=False, default=1e-3)
    parser.add_argument("-o", "--optimizer", type=str, required=False, default="adam")
    parser.add_argument("-g", "--gpus", type=int, required=False, default=4)
    args = parser.parse_args()

    if args.gpus < 1 or args.gpus > 4:
        args.gpus = 4

    torch.cuda.empty_cache()
    pl.seed_everything(42)

    num_classes = 16

    # Custom transform to fit MobileNet input format
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    trainer = None
    data_module = RVLCDIPDataModule(args.input_dir, batch_size=args.batch_size, transform=transform)
    data_module.setup(mode="image")

    model = MobileNetV3Model(pretrained=args.pretrained, in_channels=3, num_classes=num_classes, lr=args.learning_rate)

    # Use integrated Weights&Biases logger to log metadata, loss and accuracy
    wandb_logger = pl.loggers.WandbLogger(
        project="tb-experiments",
        config={
            "architecture": "MobileNetV3-Large",
            "pretrained": args.pretrained,
            "dataset": "RVL-CDIP",
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "optimizer": args.optimizer,
            "loss_function": "CrossEntropyLoss"
        },
    )

    # Callback used to save best model to disk based on "val_acc" logged value
    callbacks = [pl.callbacks.ModelCheckpoint(
                            dirpath="models/mobilenetv3_pretrained" if args.pretrained else "models/mobilenetv3_base",
                            filename="mobilenetv3_best",
                            monitor="val_acc",
                            save_weights_only=True,
                            save_last=True,
                            mode="max",
                            verbose=True)]

    if torch.cuda.is_available():
        trainer = pl.Trainer(accelerator='gpu',
                             max_epochs=args.epochs,
                             strategy="ddp_find_unused_parameters_true",
                             devices=[i for i in range(args.gpus)],
                             logger=wandb_logger,
                             callbacks=callbacks)
        print(f"Trainer is set with {args.gpus} GPUs")
    else:
        trainer = pl.Trainer(max_epochs=args.epochs,
                             logger=wandb_logger,
                             callbacks=callbacks)
        print(f"No GPU found, trainer is set on CPU")

    trainer.fit(model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.val_dataloader())
