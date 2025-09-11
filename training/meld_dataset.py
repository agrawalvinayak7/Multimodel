import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
import cv2
import numpy as np
import torch
import subprocess
import torchaudio


class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        self.data = pd.read_csv(csv_path)

        self.video_dir = video_dir

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear

        self.emotion_map = {'anger': 0, 'disgust': 1, 'sadness': 2,
            'joy': 3, 'neutral': 4, 'surprise': 5, 'fear': 6}

        self.sentiment_map = {'positive': 0, 'neutral': 1, 'negative': 2}

    def _extract_audio_features(self, video_path):
        audio_path = video_path.replace('.mp4', '.wav')

        try:
            subprocess.run([
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

                
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512
                )
            
            mel_spec = mel_spectrogram(waveform)

            #normlaize

            mel_spec = (mel_spec - mel_spec.mean())/ mel_spec.std()

            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:,:,:300]

            return mel_spec

        except subprocess.CalledProcessError as e:
            raise ValueError(f"FFmpeg Error {str(e)}")
        except Exception as e:
            raise ValueError(f"Audio Error {str(e)}")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)


    #helper function to load the video frames
    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        try:
            if not cap.isOpened():
                raise ValueError(f"video file not found at {video_path}")
            #try and read the first frame of the video, if it fails, raise an error
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Error reading video frames from {video_path}")

            #reset the index to not skip the first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while len(frames)<30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (224, 224))
                frame = frame/255.0
                frames.append(frame)

        except Exception as e:
                raise ValueError(f"Error reading video frames from {str(e)}")
        finally:
                cap.release()

                if len(frames) == 0:
                    raise ValueError(f"No frames found in {video_path}")

                #pad or truncate the frames to 30

                if len(frames) < 30:
                    frames += [np.zeros_like(frames[0])] * (30 - len(frames))
                else:
                    frames = frames[:30]

                    #before permute (frames, height, width, channels)
                    #after permute (frames, channels, height, width)
                    #doing thsi because the resnet model expects the input in this format
                return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)




    def __len__(self):
        return len(self.data)
    


    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_filename = f"""dia{row['Dialogue_ID']}_utt{
            row['Utterance_ID']}.mp4"""


        path = os.path.join(self.video_dir, video_filename)
        video_path_exist = os.path.exists(path)

        if video_path_exist == False:
            raise FileNotFoundError(f"Video file not found: {path}")
            #neural networks require fixed sizes and consistent inputs and dimensions, so we want the input to be 128, reagardless of what the utterance is, so if it is too short we will pas it 

        text_input = self.tokenizer(row['Utterance'], padding='max_length', max_length=128, truncation=True, return_tensors='pt')

        # video_frames = self._load_video_frames(path)
        audio_features = self._extract_audio_features(path)

        print(audio_features)


if __name__ == "__main__":
    meld = MELDDataset('../dataset/dev/dev_sent_emo.csv',
                       '../dataset/dev/dev_splits_complete')

    print(meld[0])