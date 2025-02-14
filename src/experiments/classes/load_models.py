import torch
import torch.nn as nn
import lightning.pytorch as pl
import time
import re
import torchmetrics
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large
from torchvision.models import resnet50
from transformers import RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
from transformers import DonutProcessor, VisionEncoderDecoderModel
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision.transforms import ToPILImage

class MultiCNNModel(pl.LightningModule):
    def __init__(self, mobilenet_checkpoint_path, resnet50_checkpoint_path, in_channels=3, num_classes=16, lr=1e-4):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.lr = lr

        self.mobilenet_checkpoint_path = mobilenet_checkpoint_path
        self.resnet_checkpoint_path = resnet50_checkpoint_path

        self.mobilenet_model = self.load_mobilenet()
        self.resnet_model = self.load_resnet()

        # Freeze the trained part of both CNN models
        for param in self.mobilenet_model.parameters():
            param.requires_grad = False
        for param in self.resnet_model.parameters():
            param.requires_grad = False

        # Add a final classification bloc combining outputs of both models
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.num_classes, 256),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(256, self.num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass')
        self.val_acc = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass')
        self.test_acc = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass')

    def load_mobilenet(self):
        # Load MobileNetV3-Large from disk
        model = MobileNetV3Model(pretrained=False, in_channels=3, num_classes=self.num_classes)
        model = model.load_from_checkpoint(self.mobilenet_checkpoint_path, map_location=torch.device('cpu'))
        return model

    def load_resnet(self):
        # Load ResNet50 from disk
        model = ResNet50Model(pretrained=False, in_channels=3, num_classes=self.num_classes)
        model = model.load_from_checkpoint(self.resnet_checkpoint_path, map_location=torch.device('cpu'))
        return model

    def forward(self, x):
        x1 = self.mobilenet_model(x)
        x2 = self.resnet_model(x)
        # Concatenate the outputs of the two models along the batch dimension for the classifier input
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc(logits, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc(logits, y), prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_acc(logits, y), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

class HybridModel(pl.LightningModule):
    def __init__(self, pretrained=True, num_classes=16, learning_rate=1e-4):
        super().__init__()
        self.num_classes = num_classes
        self.lr = learning_rate

        # Load a ResNet50 pretrained on ImageNet
        self.cnn_model = resnet50(pretrained=pretrained)

        # Load the RoBERTa model and freeze the architecture
        self.roberta_model = RobertaModel.from_pretrained("roberta-base", return_dict=True)
        for param in self.roberta_model.parameters():
            param.requires_grad = False

        # Compute concatenated number of features
        n_features = self.cnn_model.fc.in_features + self.roberta_model.config.hidden_size

        # Remove last ResNet50 layer
        self.cnn_model = nn.Sequential(*list(self.cnn_model.children())[:-1])

        # Custom final classification bloc
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.num_classes)
        )

        self.train_acc = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass')
        self.val_acc = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass')
        self.test_acc = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass')

    def forward(self, image_input, text_input):
        # Outputs of ResNet50 + RoBERTa
        cnn_output = self.cnn_model(image_input)
        cnn_output = torch.flatten(cnn_output, start_dim=1)

        # Ignore the last classification layer of RoBERTa
        text_output = self.roberta_model(**text_input).last_hidden_state[:, 0, :]

        # Concatenate the outputs and pass through the classifier bloc
        combined_output = torch.cat((cnn_output, text_output), dim=1)
        logits = self.classifier(combined_output)

        return logits

    def training_step(self, batch, batch_idx):
        image_input, input_ids, attention_mask, labels = batch

        logits = self(image_input, {'input_ids': input_ids, 'attention_mask': attention_mask})
        loss = F.cross_entropy(logits, labels)

        self.train_acc(logits.argmax(dim=1), labels)
        self.log('train_acc', self.train_acc, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image_input, input_ids, attention_mask, labels = batch

        logits = self(image_input, {'input_ids': input_ids, 'attention_mask': attention_mask})
        loss = F.cross_entropy(logits, labels)

        self.val_acc(logits.argmax(dim=1), labels)
        self.log('val_acc', self.val_acc, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class RoBERTaModel(pl.LightningModule):
    def __init__(self, num_classes=16, lr=1e-4):
        super(RoBERTaModel, self).__init__()
        self.num_classes = num_classes
        self.lr = lr

        self.test_outputs = []

        # Load the base RoBERTa model
        self.model = RobertaModel.from_pretrained("roberta-base", return_dict=True)

        # Freeze parameters that won't be trained
        for param in self.model.parameters():
            param.requires_grad = False

        # Add final layers for the classification
        self.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 786),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(786, num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass')
        self.val_acc = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass')
        self.test_acc = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass')

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.pooler_output)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return loss, logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        loss, preds = self.forward(input_ids, attention_mask, labels)

        self.train_acc(preds.argmax(dim=1), labels)
        self.log('train_loss', loss, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        loss, preds = self.forward(input_ids, attention_mask, labels)

        self.val_acc(preds.argmax(dim=1), labels)
        self.log('val_loss', loss, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)

        outputs = self.model(input_ids, attention_mask)
        embeddings = outputs.last_hidden_state

        return loss

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        loss, logits = self.forward(input_ids, attention_mask, labels)

        # Compute test data predictions and store them for future use
        for i in range(len(batch)):
            pred = torch.argmax(logits)
            self.test_outputs.append((labels.item(), pred.item()))

        self.test_acc(logits.argmax(dim=1), labels)
        self.log('test_acc', self.test_acc, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

class MobileNetV3Model(pl.LightningModule):
    def __init__(self, pretrained=True, in_channels=3, num_classes=16, lr=1e-4):
        super(MobileNetV3Model, self).__init__()
        self.supported_optimizers = ["adam", "sgd", "rmsprop", "adagrad"]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.lr = lr

        self.test_outputs = []

        # Load MobileNetV3 model. Using weights of ImageNet are indicated in the class constructor
        self.model = mobilenet_v3_large(pretrained=pretrained, progress=True)

        # Remove the final layer to add a custom one for the RVL-CDIP dataset
        num_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = torch.nn.Linear(num_features, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass')
        self.val_acc = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass')
        self.test_acc = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        preds = self.model(x)

        loss = self.loss_fn(preds, y)
        self.train_acc(torch.argmax(preds, dim=1), y)

        self.log('train_loss', loss.item(), on_epoch=True, logger=True, sync_dist=True)
        self.log('train_acc', self.train_acc, on_epoch=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        preds = self.model(x)

        loss = self.loss_fn(preds, y)
        self.val_acc(torch.argmax(preds, dim=1), y)

        self.log('val_loss', loss.item(), on_epoch=True, logger=True, sync_dist=True)
        self.log('val_acc', self.val_acc, on_epoch=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        s = time.time()
        x, y = batch
        preds = self.model(x)
        self.test_acc(torch.argmax(preds, dim=1), y)

        # Store prediction for future use
        for i in range(len(batch)):
            pred = torch.argmax(preds)
            self.test_outputs.append((y.item(), pred.item()))

        self.log('test_acc', self.test_acc, on_epoch=True)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        return [optimizer], [scheduler]

class ResNet50Model(pl.LightningModule):
    def __init__(self, pretrained=True, in_channels=3, num_classes=16, lr=1e-4):
        super(ResNet50Model, self).__init__()
        self.supported_optimizers = ["adam", "sgd", "rmsprop", "adagrad"]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.lr = lr

        self.model = resnet50(pretrained=pretrained)

        # Add custom final layers to the model to match with the RVL-CDIP dataset
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 128),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()

        self.test_outputs = []

        self.train_acc = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass')
        self.val_acc = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass')
        self.test_acc = torchmetrics.Accuracy(num_classes=num_classes, task='multiclass')

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch

        preds = self.model(x)

        loss = self.loss_fn(preds, y)
        self.train_acc(torch.argmax(preds, dim=1), y)

        self.log('train_loss', loss.item(), on_epoch=True, logger=True, sync_dist=True)
        self.log('train_acc', self.train_acc, on_epoch=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        preds = self.model(x)

        loss = self.loss_fn(preds, y)
        self.val_acc(torch.argmax(preds, dim=1), y)

        self.log('val_loss', loss.item(), on_epoch=True, logger=True, sync_dist=True)
        self.log('val_acc', self.val_acc, on_epoch=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        s = time.time()

        x, y = batch
        preds = self.model(x)
        self.test_acc(torch.argmax(preds, dim=1), y)

        # Store label and prediction for each unique data in batch
        for i in range(len(batch)):
            pred = torch.argmax(preds)
            self.test_outputs.append((y.item(), pred.item()))

        self.log('test_acc', self.test_acc, on_epoch=True)

class PretrainedModel:
    def __init__(self, model):
        # This class is supposed to be used for Donut or Dit model. This method initialize the corresponding
        # model and processor
        if model == "donut":
            print("Using Donut model for inference...")
            self.processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
            self.model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
            self.model.eval()
        elif model == "dit":
            print("Using DiT model for inference...")
            self.processor = AutoImageProcessor.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
            self.model = AutoModelForImageClassification.from_pretrained("microsoft/dit-base-finetuned-rvlcdip")
        self.model_name = model

        # Inference is done on CPU
        self.device = "cpu"

    def infer_model(self, batch):
        if self.model_name == "donut":
            # Prepare prompt and tokenizer for Donut model
            task_prompt = "<s_rvlcdip>"
            decoder_input_ids = self.processor.tokenizer(task_prompt, add_special_tokens=False,
                                                    return_tensors="pt").input_ids

            s = time.time()
            inputs = batch[0]
            targets = batch[1]

            # Convert data tensor to PIL image (works if batch size if set to 1)
            image = ToPILImage()(inputs[0])

            pixel_values = self.processor(image, return_tensors="pt").pixel_values

            # Generate output with basic parameters provided in the HuggingFace documentation
            outputs = self.model.generate(
                pixel_values.to(self.device),
                decoder_input_ids=decoder_input_ids.to(self.device),
                max_length=self.model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

            # Decode and parse model output to get predicted RVL-CDIP class
            sequence = self.processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

            duration = round(time.time() - s, 2)
            # Return the predicted class number and target class number with the inference duration
            return self.processor.token2json(sequence)["class"], targets[0].item(), duration

        elif self.model_name == "dit":
            s = time.time()
            inputs = batch[0]
            targets = batch[1]

            image = ToPILImage()(inputs[0])

            # Use the input processor for DiT and process the input
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)

            logits = outputs.logits

            # The model predicts one of the 16 RVL-CDIP classes, so I retrieve the max probability one
            predicted_class_idx = logits.argmax(-1).item()

            # Use the model integrated dict to get the class name
            predicted_class = self.model.config.id2label[predicted_class_idx]

            # Parse the class name to remove whitespace and match the class names of the dataset
            predicted_class = predicted_class.replace(" ", "_")
            duration = round(time.time() - s, 2)

            # Return also the predicted class, target class and inference duration
            return predicted_class, targets[0].item(), duration


