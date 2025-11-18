# README

# Multimodal Video Sentiment Analysis SaaS

A full-stack AI application that analyzes video files to detect **Emotions** and **Sentiments**. By using a "Multimodal" approach, this application doesn't just look at one thing—it combines visual cues (face expressions), audio (tone of voice), and text (transcription) to understand the true feeling behind a video.

## 🚀 Key Features

- **Multimodal AI:** Analyzes Video, Audio, and Text simultaneously for higher accuracy.
- **Granular Analysis:** Breaks down videos into specific utterances (sentences) and analyzes each one.
- **7 Emotion Classes:** Detects Anger, Disgust, Fear, Joy, Neutral, Sadness, and Surprise.
- **Sentiment Detection:** Classifies content as Positive, Neutral, or Negative.
- **Developer API:** Provides a secure API with quota management for developers to integrate analysis into their own apps.
- **Modern Dashboard:** A clean Next.js user interface to upload videos and view detailed results.

## 🛠️ Tech Stack

### Frontend (Web App)

- **Framework:** [Next.js](https://nextjs.org/) (App Router)
- **Styling:** Tailwind CSS
- **Authentication:** NextAuth.js (Secure login/signup)
- **Database:** PostgreSQL (via Prisma ORM)
- **Language:** TypeScript

### Backend (AI & Infrastructure)

- **Machine Learning:** PyTorch
- **Speech-to-Text:** OpenAI Whisper
- **Cloud Computing:** AWS SageMaker (for running the AI models)
- **Storage:** AWS S3 (for storing uploaded videos)
- **Video Processing:** FFmpeg & OpenCV

## 🧠 How the AI Works (The "Multimodal" Part)

Most sentiment analysis tools only read text. This project is different because it uses three separate "Encoders":

1. **Text Encoder (BERT):** Reads the words spoken in the video to understand the meaning.
2. **Video Encoder (R3D_18):** Looks at the video frames to recognize facial expressions and body language.
3. **Audio Encoder (CNN):** Listens to the sound waves (spectrograms) to detect tone and pitch.

These three signals are sent to a **Fusion Layer**, which combines them into a single prediction. This mimics how humans naturally understand emotion.

## ⚙️ Installation & Setup

### Prerequisites

- **Node.js** (v18 or higher)
- **Python** (v3.11) - [Download Here](https://www.python.org/downloads/)
- **PostgreSQL Database**
- **AWS Account** with the following configured:
    - **S3 Bucket** (for storing datasets and videos)
    - **SageMaker Quota:** Request a quota increase for **Training Job usage** (Instance type: `ml.g5.xlarge` recommended).

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/multimodel.git
cd multimodel
```

### 2. Backend Setup & Data Preparation

Navigate to the backend folder and install the required Python libraries.

```bash
pip install -r backend/training/requirements.txt
```

**Download the Dataset:**

1. Visit the [MELD Dataset](https://affective-meld.github.io/) page.
2. Download and extract the dataset.
3. Place the extracted files into the `backend/dataset` directory.

### 3. AWS Configuration (Crucial Step)

To train and deploy the model, you must set up specific permissions in AWS.

### **A. S3 Bucket Setup**

Create an S3 bucket (e.g., `video-sentimental-analysis`) and upload your `dataset` folder there.

### **B. Create Training Role (IAM)**

Create a role in AWS IAM for SageMaker training with the following permissions:

1. **AmazonSageMakerFullAccess**
2. **S3 Access Policy:** Add this inline policy to allow access to your dataset bucket:

```json
{
	"Version": "2012-10-17",
	"Statement": [
		{
			"Sid": "VisualEditor0",
			"Effect": "Allow",
			"Action": [
				"s3:PutObject",
				"s3:GetObject",
				"s3:ListBucket",
				"s3:DeleteObject"
			],
			"Resource": [
				"arn:aws:s3:::your-bucket-name",
				"arn:aws:s3:::your-bucket-name/*"
			]
		}
	]
}
```

### **C. Start Training Job**

Update the `backend/train_sagemaker.py` file with your S3 bucket name and the Role ARN you just created. Then run:

```bash
python backend/train_sagemaker.py
```

### **D. Create Deployment Role**

Create a new role for deploying the endpoint with these permissions:

1. **AmazonSageMakerFullAccess**
2. **CloudWatchLogsFullAccess**
3. **S3 Access Policy** (Use the same JSON policy as above).

### **E. Deploy Endpoint**

After training, put your model file (`model.tar.gz`) in your S3 bucket. Update `backend/deployment/deploy_endpoint.py` with your model path and Deployment Role ARN. Run:

```bash
python backend/deployment/deploy_endpoint.py
```

### **E. Deploy Endpoint**

After training, put your model file (`model.tar.gz`) in your S3 bucket. Update `backend/deployment/deploy_endpoint.py` with your model path and Deployment Role ARN. Run:

```bash
python backend/deployment/deploy_endpoint.py
```

### **F. Create Invocation User**

For the Frontend to talk to the Backend, create a user in IAM with specific permissions to invoke the endpoint. **These are the keys you will use in your `.env` file.**

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject"
            ],
            "Resource": [
                "arn:aws:s3:::sentiment-analysis-saas/inference/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:InvokeEndpoint"
            ],
            "Resource": [
                "arn:aws:sagemaker:us-east-1:YOUR_ACCOUNT_ID:endpoint/sentiment-analysis-endpoint"
            ]
        }
    ]
}
```

### 4. Frontend Setup

Navigate to the frontend folder and install dependencies:

```bash
cd frontend
npm install
```

Create a `.env` file in the `frontend` directory based on `.env.example` and populate it with the keys from the **Invocation User** created above:

```bash
# Database connection
DATABASE_URL="your-database-url"

# NextAuth Secret (Generate one using `npx auth secret`)
AUTH_SECRET="your-super-secret-key"

# AWS Configuration
AWS_REGION="your-region"
AWS_ACCESS_KEY_ID="your-access-key-from-step-3F"
AWS_SECRET_ACCESS_KEY="your-secret-key-from-step-3F"
AWS_INFERENCE_BUCKET="your-s3-bucket-name"
AWS_ENDPOINT_NAME="sentiment-analysis-endpoint"
```

Initialize the database:

```bash
npx prisma db push
```

Run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](https://www.google.com/search?q=http://localhost:3000) in your browser.

## 🔌 API Usage

Developers can use the API to analyze videos programmatically. You can find your **Secret Key** in the dashboard after logging in.

**Example Request (cURL):**

Bash

```bash
# 1. Get Upload URL
curl -X POST \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"fileType": ".mp4"}' \
  http://localhost:3000/api/upload-url

# 2. Upload your video to the returned URL...

# 3. Trigger Analysis
curl -X POST \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"key": "file_key_from_step_1"}' \
  http://localhost:3000/api/sentiment-inference
```

## 📂 Project Structure

- `frontend/`: Contains the Next.js website, UI components, and API routes.
    - `src/app/`: Pages and routing.
    - `src/components/`: Reusable UI elements (Upload button, Charts).
    - `prisma/`: Database schema.
- `backend/`: Contains the Python code for the AI.
    - `training/`: Scripts to train the neural network (`train.py`, `models.py`).
    - `deployment/`: Scripts to deploy the model to AWS SageMaker (`deploy_endpoint.py`).
    - `dataset/`: Logic for handling the MELD dataset.