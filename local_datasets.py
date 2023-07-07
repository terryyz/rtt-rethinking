from datasets import load_dataset
from utils.constants import *


def read_data(dataset_name, src_lang, tgt_lang, split):
    if dataset_name == "wmt20-news":
        src_data, tgt_data = load_wmt20_news_dataset(src_lang, tgt_lang, split)
    elif dataset_name == "wmt20-bio":
        src_data, tgt_data = load_wmt20_bio_dataset(src_lang, tgt_lang, split)
    elif dataset_name == "flores_101":
        src_data, tgt_data = load_flores_dataset(src_lang, tgt_lang, split)
    else:
        raise KeyError("Invalid Dataset Name!")
    return src_data, tgt_data


def load_wmt20_news_dataset(src, tgt, split):
    if split == "devtest":
        with open(f"data/wmt20/sources/newstest2020-{src}{tgt}-src.{src}.txt") as f:
            src_data = f.read().splitlines()
        with open(f"data/wmt20/references/newstest2020-{src}{tgt}-ref.{tgt}.txt") as f:
            tgt_data = f.read().splitlines()
        return src_data, tgt_data
    else:
        return [],[]

def load_wmt20_bio_dataset(src, tgt, split):
    if split == "devtest":
        with open(f"data/wmt20bio/sources/{src}2{tgt}_{src}.txt") as f:
            src_data = f.read().splitlines()
        with open(f"data/wmt20bio/references/{src}2{tgt}_{tgt}.txt") as f:
            tgt_data = f.read().splitlines()
        return src_data, tgt_data
    else:
        return [],[]

def load_flores_dataset(src, tgt, split):
    src_dataset = load_dataset("gsarti/flores_101", FLORES_M2M_DICT[src], cache_dir=f"./data")
    tgt_dataset = load_dataset("gsarti/flores_101", FLORES_M2M_DICT[tgt], cache_dir=f"./data")
    return [pair for pair in src_dataset[split]['sentence']], \
           [pair for pair in tgt_dataset[split]['sentence']]
