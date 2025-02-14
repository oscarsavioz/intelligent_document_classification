from classes.load_models import RoBERTaModel
import torch
import lightning.pytorch as pl
import argparse
import warnings
from classes.datasets import RVLCDIPDataModule
import os

os.environ['TORCH_USE_CUDA_DSA'] = '1'

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True)
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

    trainer = None
    data_module = RVLCDIPDataModule(args.input_dir, batch_size=args.batch_size)
    data_module.setup(mode="text")

    print(f"Initialized train dataset with {data_module.dataset_size()[0]} data")
    print(f"Initialized validation dataset with {data_module.dataset_size()[1]} data")
    print(f"Initialized test dataset with {data_module.dataset_size()[2]} data")

    model = RoBERTaModel(num_classes=num_classes, lr=args.learning_rate)

    # Use integrated Weights&Biases logger to log metadata, loss and accuracy
    wandb_logger = pl.loggers.WandbLogger(
        project="tb-experiments",
        config={
            "architecture": "RoBERTa",
            "dataset": "RVL-CDIP",
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "optimizer": args.optimizer,
            "loss_function": "CrossEntropyLoss"
        },
    )

    # Callback used to save best model to disk based on "val_acc" logged value
    callbacks = [pl.callbacks.ModelCheckpoint(
                            dirpath="models/roberta",
                            filename="roberta_best",
                            monitor="val_acc",
                            save_weights_only=True,
                            save_last=True,
                            mode="max",
                            verbose=True)]

    if torch.cuda.is_available():
        trainer = pl.Trainer(accelerator='gpu',
                             strategy="ddp",
                             max_epochs=args.epochs,
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
