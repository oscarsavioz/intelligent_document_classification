from classes.datasets import RVLCDIPDataModule
from classes.load_models import ResNet50Model, MobileNetV3Model, RoBERTaModel
from torchvision.transforms import transforms
import torch
import lightning.pytorch as pl
import argparse
import warnings
import csv

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    # Should be in ["resnet50", "mobilenetv3_pretrained" "mobilenetv3_base", "roberta"]
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-b", "--batch_size", type=int, required=True)
    args = parser.parse_args()

    torch.cuda.empty_cache()
    pl.seed_everything(42)

    num_classes = 16

    # Custom transform to fit ResNet50 input format
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    trainer = None
    model = None

    data_module = RVLCDIPDataModule(args.input_dir, batch_size=args.batch_size, transform=transform)
    data_module.setup(mode="image", stage="test")

    if args.model == "resnet50":
        print("Using ResNet50")
        try:
            model = ResNet50Model(pretrained=False, in_channels=3, num_classes=num_classes)
            model = model.load_from_checkpoint("models/resnet50/resnet50_best.ckpt", map_location=torch.device('cpu'))
        except:
            print("Error loading pre-trained ResNet50 model, please verify that resnet50_best.ckpt exsits")
    elif args.model == "mobilenetv3_pretrained":
        print("Using MobileNetV3 - Pretrained")
        try:
            model = MobileNetV3Model(pretrained=True, in_channels=3, num_classes=num_classes)
            model = model.load_from_checkpoint("models/mobilenetv3_pretrained/mobilenetv3_best.ckpt", map_location=torch.device('cpu'))
            print(model)
        except:
            print("Error loading pre-trained MobileNetV3-Pretrained model, please verify that mobilenetv3_best.ckpt exsits")
    elif args.model == "mobilenetv3_base":
        print("Using MobileNetV3 - Base")
        model = MobileNetV3Model(pretrained=False, in_channels=3, num_classes=num_classes)
        model = model.load_from_checkpoint("models/mobilenetv3_base/mobilenetv3_best.ckpt", map_location=torch.device('cpu'))
        print(model)
    elif args.model == "roberta":
        print("Using RoBERTa")
        model = RoBERTaModel.load_from_checkpoint("models/roberta/roberta_best.ckpt", map_location=torch.device('cpu'))
    else:
        print("Unsupported model for inference")

    print(f"Nombre de données : {len(data_module.test_dataloader())}")
    trainer = pl.Trainer(accelerator="cpu")
    trainer.test(model, dataloaders=data_module.test_dataloader())

    with open('tests/roberta_out.txt', 'w', newline='') as f:
        for output in model.test_outputs:
            f.write(f"{output[0]},{output[1]}")
            f.write("\n")
