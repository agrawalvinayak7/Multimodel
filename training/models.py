import torch
import torch.nn as nn
from transformers import BertModel
from torchvision.models.video import r3d_18

from meld_dataset import MELDDataset


class text_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # defining what we want to use i.e bert model (pretrained model)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # freezing the bert model, so we don't train it
        for param in self.bert.parameters():
            param.requires_grad = False
        # linear layer to project the bert embeddings to 128 dimmensions
        self.projection = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        # extract bert embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # use [CLS] token representation, bert can train the cls token to capture the meaning of the entire sentence
        # so this cls token which will be pooler output will act as the summary of the full text
        pooler_output = outputs.pooler_output

        return self.projection(pooler_output)


class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # this will get the pretrained model from torchvision which will be r3d_18
        self.backbone = r3d_18(pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False

        num_fts = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_fts, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        # [batchsize, frames, channels , height, width] -> [batchsize, channels, frames, height, width]
        x = x.transpose(1, 2)
        return self.backbone(x)


class Audio_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # here we detect patterns usnig the convulutional netwrok
        self.conv_layers = nn.Sequential(
            # lower level features
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # higher level features
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        for param in self.conv_layers.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)

        )

    def forward(self, x):
        # [batchsize, channels (1), 64 mel freq bins, 300 time steps]
        # here we remove the single channel dimmension that will be the part of the spectrogram
        # for example input shape: torch.size([2, 1, 64, 300])
        # squeezed shape : torch.size([2, 64, 300])
        # so now the thing that is happening here is that
        # 300 is the time component which is plotted along the x axis, and 64 is the freq which is plotted along the y axis,
        # the conv1d layer will then go through the x axis and then find patterns in the y axis, and this format is req because it doesnot know
        # how to handle the 1 in between batch size and freq
        x = x.squeeze(1)

        features = self.conv_layers(x)
        # features output : [batch_size, 128, 1]
        # because we are using adaptive average pool and hence 1

        return self.projection(features.squeeze(-1))


class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()

        # ecoders
        self.text_encoder = text_encoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = Audio_encoder()

        # fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(128*3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # classification heads

        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7)
        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )

    def forward(self, text_inputs, video_frames, audio_features):
        text_features = self.text_encoder(
            text_inputs['input_ids'],
            text_inputs['attention_mask']
        )
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)

        # concatenate multimodel features

        combined_features = torch.cat([
            text_features,
            video_features,
            audio_features
        ], dim=1)

        # batch size, 128*3

        fused_features = self.fusion_layer(combined_features)

        emotion_output = self.emotion_classifier(fused_features)
        sentiment_output = self.sentiment_classifier(fused_features)

        return {
            'emotions': emotion_output,
            'sentiment': sentiment_output
        }


if __name__ == "__main__":
    dataset = MELDDataset(
        '../dataset/train/train_sent_emo.csv', '../dataset/train/train_splits')

    sample = dataset[0]

    model = MultimodalSentimentModel()
    model.eval()

    text_inputs = {
        'input_ids': sample['text_inputs']['input_ids'].unsqueeze(0),
        'attention_mask': sample['text_inputs']['attention_mask'].unsqueeze(0)
    }
    video_frames = sample['video_frames'].unsqueeze(0)
    audio_features = sample['audio_features'].unsqueeze(0)

    with torch.inference_mode():
        outputs = model(text_inputs, video_frames, audio_features)

        emotion_probs = torch.softmax(outputs['emotions'], dim=1)[0]
        sentiment_probs = torch.softmax(outputs['sentiment'], dim=1)[0]

    emotion_map = {
        0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'neutral', 5: 'sadness', 6: 'surprise'
    }

    sentiment_map = {
        0: 'negative', 1: 'neutral', 2: 'positive'
    }


    for i, prob in enumerate(emotion_probs):
        print(f"{emotion_map[i]}: {prob:.2f}")

    for i, prob in enumerate(sentiment_probs):
        print(f"{sentiment_map[i]}: {prob:.2f}")

    print("Predictions for utterance")

