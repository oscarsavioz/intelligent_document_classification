from classes.load_models import PretrainedModel
from torchvision.transforms import transforms
import torch
import lightning.pytorch as pl
import argparse
import warnings
import sys
from classes.utils import class_to_index_dict
from classes.datasets import RVLCDIPDataModule

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-o", "--output_folder", type=str, required=True)
    args = parser.parse_args()

    supported_models = ["dit", "donut"]
    batch_size = 1
    num_classes = 16

    class_to_index = class_to_index_dict()

    # Simple transform for Donut and Dit, no need to resize image
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if args.model not in supported_models:
        print(f"Model {args.model} not supported, please choose between {supported_models}")
        sys.exit(0)

    model = PretrainedModel(args.model)

    data_module = RVLCDIPDataModule(args.input_dir, batch_size, transform)
    data_module.setup(mode="image", stage="test")

    print(f"Dataset ready with {len(data_module.test_dataset)} data")

    model_outputs = []
    total_duration = 0.0
    n_data = 1

    for batch in data_module.test_dataloader():
        # Infer model to get predicted class and target class
        pred, target, duration = model.infer_model(batch)
        model_outputs.append((class_to_index[pred], target))
        total_duration += duration

        print(f"Processed data n°{n_data} in {duration} s.")
        n_data += 1

    print(f"Inference on {args.model} done with a mean of {round(total_duration/len(data_module.test_dataset), 2)} s. per image")

    # Write results to a text file for further uses
    with open(f'{args.output_folder}/{args.model}_out.txt', 'w', newline='') as f:
        for output in model_outputs:
            f.write(f"{output[0]},{output[1]}")
            f.write("\n")

