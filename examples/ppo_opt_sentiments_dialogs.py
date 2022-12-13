# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function

from datasets import load_dataset
from transformers import pipeline, AutoTokenizer
import os
import yaml

import trlx
import torch
from functools import partial
from typing import List
from trlx.data.configs import TRLConfig


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


default_config = yaml.safe_load(open("configs/ppo_opt_dialogs.yml"))


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)
    tokenizer = AutoTokenizer.from_pretrained(default_config["model"]["tokenizer_path"])

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=device,
    )

    def reward_fn(tokenizer, samples: List[str]) -> List[float]:
        samples = [s.split(tokenizer.eos_token)[-1] for s in samples]
        sentiments = list(map(get_positive_score, sentiment_fn(samples)))
        return sentiments
    
    dataset = load_dataset("bavard/personachat_truecased")
    
    def chunking(examples):
        inputs = [
            tokenizer.eos_token.join(history) + tokenizer.eos_token
            for history in examples["history"]
        ]
        return {"chunks": inputs}

    tokenized_datasets = (
        dataset.map(chunking, batched=True, remove_columns=dataset["train"].column_names)
    )

    # Take few words off of movies reviews as prompts
    train_prompts = [e["chunks"] for e in tokenized_datasets["train"]]
    eval_prompts = [e["chunks"] for e in tokenized_datasets["validation"]][:128]

    model = trlx.train(
        reward_fn=partial(reward_fn, tokenizer),
        prompts=train_prompts,
        eval_prompts=eval_prompts,
        config=config,
    )


if __name__ == "__main__":
    main()
