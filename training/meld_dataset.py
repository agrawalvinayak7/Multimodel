import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import os
class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        self.data = pd.read_csv(csv_path)

        self.video_dir = video_dir

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    #Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear
        
        self.emotion_map = {'anger':0, 'disgust':1, 'sadness':2, 'joy':3, 'neutral':4, 'surprise':5, 'fear':6}
        
        self.sentiment_map = {'positive':0, 'neutral':1, 'negative':2}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_filname = f"""dia{row['Dialog_ID']}
        _utt{row['Utterance_ID']}.mp4"""


        path = os.path.join(self.video_dir, video_filname)
if __name__ == "__main__":
    meld = MELDDataset('../dataset/dev/dev_sent_emo.csv',
                       '../dataset/dev/dev_splits_complete')