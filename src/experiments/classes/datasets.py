import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import pytorch_lightning as pl
from torchvision import transforms
import random
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, AutoTokenizer, AdamW, get_cosine_schedule_with_warmup
import csv

class RVLCDIPDataset(Dataset):
    def __init__(self, mode, data_dir, label_file, transform, tokenizer=None):
        if mode not in ["image", "text", "hybrid"]:
            self.mode = "image"
        else:
            self.mode = mode

        # Root directory of RVL-CDIP dataset
        self.data_dir = data_dir
        self.label_file = label_file
        self.data = self.load_data()
        self.tokenizer = tokenizer
        self.transform = transform

    def load_data(self):
        data = []
        count = 0
        # Data are prepared by a large list of tuples, each tuple contains the image path and the corresponding class number
        with open(self.label_file, 'r') as file:
            for line in file:
                image_path, label = line.strip().split(' ')
                if self.mode == "image":
                    # Data contains image path and corresponding target class
                    data.append((os.path.join(self.data_dir, 'images', image_path), int(label)))
                elif self.mode == "text" or self.mode == "hybrid":
                    textfile_path = os.path.join(self.data_dir, 'images', os.path.join(
                        os.path.dirname(image_path), "ocr.txt"))
                    if os.path.exists(textfile_path):
                        if self.mode == "hybrid":
                            # Data contains image path, textfile path and corresponding target class
                            data.append((os.path.join(self.data_dir, 'images', image_path), textfile_path, int(label)))
                        else:
                            # Data contains textfile path and corresponding target class
                            data.append((textfile_path, int(label)))
        random.shuffle(data)

        return data

    def get_filetext_encoding(self, filepath):
        # Read data ocr text file
        with open(filepath, 'r') as f:
            text = f.read()

        # Encode textual content using RoBERTa tokenizer
        encoding = self.tokenizer.encode_plus(text,
                                              truncation=True,
                                              add_special_tokens=True,
                                              max_length=256,
                                              padding='max_length',
                                              return_attention_mask=True,
                                              return_tensors='pt')

        # Return correct data for RoBERTa model
        return encoding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Depending on used mode, return corresponding data (image or/and text file encoding for roberta)
        if self.mode == "image":
            while True:
                try:
                    image = Image.open(self.data[idx][0]).convert('RGB')
                    if self.transform is not None:
                        image = self.transform(image)
                    return image, torch.tensor(self.data[idx][1])
                except:
                    # If actual image can't be read, abuse by choosing a random index in the dataset
                    idx = random.randint(0, len(self.data) - 1)
        elif self.mode == "text":
            while True:
                try:
                    encoding = self.get_filetext_encoding(self.data[idx][0])

                    return encoding['input_ids'].view(-1), encoding['attention_mask'].view(-1), torch.tensor(self.data[idx][1])
                except:
                    # If actual data can't be read, abuse by choosing a random index in the dataset
                    idx = random.randint(0, len(self.data) - 1)
        elif self.mode == "hybrid":
            while True:
                try:
                    image = Image.open(self.data[idx][0]).convert('RGB')
                    if self.transform is not None:
                        image = self.transform(image)

                    encoding = self.get_filetext_encoding(self.data[idx][1])
                    # Return image and roberta encodings for the text file
                    return image, encoding['input_ids'].view(-1), encoding['attention_mask'].view(-1), torch.tensor(self.data[idx][2])
                except:
                    # If actual data can't be read, abuse by choosing a random index in the dataset
                    idx = random.randint(0, len(self.data) - 1)

class RVLCDIPDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
        # The only tokenizer used is the one for RoBERTa
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        self.train_dataset = self.test_dataset = self.val_dataset = None


    def setup(self, mode="image", stage=None):
        if stage == 'fit' or stage is None:
            train_file = os.path.join(self.data_dir, 'labels', 'train.txt')
            val_file = os.path.join(self.data_dir, 'labels', 'val.txt')
            test_file = os.path.join(self.data_dir, 'labels', 'test.txt')

            self.train_dataset = RVLCDIPDataset(mode, self.data_dir, train_file, self.transform, self.tokenizer)
            self.val_dataset = RVLCDIPDataset(mode, self.data_dir, val_file, self.transform, self.tokenizer)
            self.test_dataset = RVLCDIPDataset(mode, self.data_dir, test_file, self.transform, self.tokenizer)

        # Only prepare the test datasetset when test mode is set
        elif stage == "test":
            test_file = os.path.join(self.data_dir, 'labels', 'test.txt')
            self.test_dataset = RVLCDIPDataset(mode, self.data_dir, test_file, self.transform, self.tokenizer)

    def dataset_size(self):
        return len(self.train_dataset), len(self.val_dataset), len(self.test_dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False, drop_last=True)
