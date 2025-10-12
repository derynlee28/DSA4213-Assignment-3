# Irony detection with DistilBERT and RobERTA, using full fine-tuning and LoRA

This repo reproduces irony detection experiments on the TweetEval Irony dataset.
We compare full fine-tuning and LoRA fine-tuning for DistilBERT and RoBERTa models.

## Installation
1. Clone the repository:

2. Create python environment

python3 -m venv venv
source venv/bin/activate

3. Instal dependencies:

pip install -r requirements.txt


## Run experiments

python main.py


## Dataset 

Using [TweetEval](https://huggingface.co/datasets/tweet_eval) irony dataset.  
The dataset is automatically downloaded via Hugging Face's datasets library.