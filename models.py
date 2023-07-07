import os
import torch
import fairseq
from tqdm import tqdm
from easynmt import EasyNMT
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, MarianMTModel, MarianTokenizer, \
    M2M100ForConditionalGeneration, M2M100Tokenizer

def load_model(model_name, cuda):
    if model_name == "mbart50-m2m":
        model = EasyNMT('mbart50_m2m', device=f'cuda:{cuda}', cache_folder="./tmp_model")

    elif model_name == "m2m-100-base":
        model = EasyNMT('m2m_100_418M', device=f'cuda:{cuda}', cache_folder="./tmp_model")

    elif model_name == "m2m-100-large":
        model = EasyNMT('m2m_100_1.2B', device=f'cuda:{cuda}', cache_folder="./tmp_model")

    elif model_name == "fair-wmt20":
        torch.hub.set_dir('tmp_model')
        model = dict()
        print(torch.__version__)
        print(torch.hub.get_dir())
        print(torch.hub.list('pytorch/fairseq'))
        model['en-ta'] = torch.hub.load('pytorch/fairseq', 'transformer.wmt20.en-ta').to(f'cuda:{cuda}')
        model['ta-en'] = torch.hub.load('pytorch/fairseq', 'transformer.wmt20.ta-en').to(f'cuda:{cuda}')
        tokenizer = None
        
    elif model_name == "Helsinki-NLP":
        model = EasyNMT('opus-mt', device=f'cuda:{cuda}', cache_folder="./tmp_model")

    elif "m2m-124":
        os.system("sh fairseq_m2m_124_data.sh")

    else:
        raise ValueError("Wrong Model Name!")

    return model


def model_translate(model, texts, src, tgt):
    results = []
    # if isinstance(model, MBartForConditionalGeneration) or isinstance(model, M2M100ForConditionalGeneration):
    #     src_name = [k for k in tokenizer.lang_code_to_id.keys() if k.split("_")[0] == src][0]
    #     tgt_name = [k for k in tokenizer.lang_code_to_id.keys() if k.split("_")[0] == tgt][0]
    #     tokenizer.src_lang = src_name
    #     for text in tqdm(texts):
    #         encoded = tokenizer(text, return_tensors="pt").to(f'cuda:{cuda}')
    #         generated_tokens = model.generate(
    #             **encoded,
    #             forced_bos_token_id=tokenizer.lang_code_to_id[tgt_name]
    #         )
    #         results.append(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
    #         torch.cuda.empty_cache()
    # elif isinstance(model, MarianMTModel):
    #     for text in tqdm(texts):
    #         encoded = tokenizer(text, return_tensors="pt").to(f'cuda:{cuda}')
    #         generated_tokens = model.generate(
    #             **encoded
    #         )
    #         results.append(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
    #         torch.cuda.empty_cache()
    # elif \
    if isinstance(model, EasyNMT):
        for i,text in enumerate(tqdm(texts)):
            if i == 0:
                a = model.translate(text, source_lang=src, target_lang=tgt)
                print(a)
                results.append(a)
            else:
                results.append(model.translate(text, source_lang=src, target_lang=tgt))
            torch.cuda.empty_cache()
    return results