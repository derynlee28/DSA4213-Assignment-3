# Irony detection with DistilBERT and RobERTA, using full fine-tuning and LoRA

This repo reproduces irony detection experiments on the TweetEval Irony dataset.
We compare full fine-tuning and LoRA fine-tuning for DistilBERT and RoBERTa models.

## Installation
1. Clone the repository:
   
git clone https://github.com/derynlee28/DSA4213-Assignment-3.git

cd DSA4213-Assignment-3

3. Create python environment
   
python3 -m venv venv

source venv/bin/activate

5. Install dependencies:
   
pip install -r requirements.txt


## Run experiments
To reproduce experiement, run:

python main.py

To view code notebook, open:

assignment_3.ipynb

## Dataset 

Using [TweetEval](https://huggingface.co/datasets/tweet_eval) irony dataset.  
The dataset is automatically downloaded via Hugging Face's datasets library.
