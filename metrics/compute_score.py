from sacrebleu.metrics import BLEU, CHRF, TER
from bert_score import score
from numpy import mean
import math 

'''
https://github.com/mjpost/sacrebleu
'''
def compute_bleu(decoded_preds,
                  decoded_labels,
                  lang,
                  corpus_level=False):
    result = dict()
    if corpus_level:
        for i in range(1,5):
            bleu = BLEU(max_ngram_order=i)
            result[f'bleu_max{i}'] = bleu.corpus_score(decoded_preds, [decoded_labels])
    else:
        for i in range(1,5):
            for pred, label in zip(decoded_preds, decoded_labels):
                bleu = BLEU(max_ngram_order=i)
                try:
                    result[f'bleu_max{i}'].append(bleu.corpus_score([pred], [[label]]))
                except:
                    result[f'bleu_max{i}'] = [bleu.corpus_score([pred], [[label]])]
    return result

def compute_spbleu(decoded_preds,
                    decoded_labels,
                    lang,
                    corpus_level=False):
    result = dict()
    if corpus_level:
        for i in range(1,5):
            bleu = BLEU(tokenize="spm", max_ngram_order=i)
            result[f'spbleu_max{i}'] = bleu.corpus_score(decoded_preds, [decoded_labels])
    else:
        for i in range(1,5):
            for pred, label in zip(decoded_preds, decoded_labels):
                bleu = BLEU(tokenize="spm", max_ngram_order=i)
                try:
                    result[f'spbleu_max{i}'].append(bleu.corpus_score([pred], [[label]]))
                except:
                    result[f'spbleu_max{i}'] = [bleu.corpus_score([pred], [[label]])]
    return result

def compute_avg_1_spbleu(decoded_preds,
                    decoded_labels,
                    lang,
                    corpus_level=False):
    result = dict()
    if corpus_level:
        for i in range(4,5):
            bleu = BLEU(tokenize="spm", max_ngram_order=i)
            tmp = []
            for j in range(math.ceil(len(decoded_preds)/1)):
                tmp.append(bleu.corpus_score(decoded_preds[j*1:(j+1)*1], [decoded_labels[j*1:(j+1)*1]]).score)
            result[f'spbleu_max{i}'] = {"score":mean(tmp)}
    else:
        for i in range(1,5):
            for pred, label in zip(decoded_preds, decoded_labels):
                bleu = BLEU(tokenize="spm", max_ngram_order=i)
                try:
                    result[f'spbleu_max{i}'].append(bleu.corpus_score([pred], [[label]]))
                except:
                    result[f'spbleu_max{i}'] = [bleu.corpus_score([pred], [[label]])]
    return result

def compute_avg_10_spbleu(decoded_preds,
                    decoded_labels,
                    lang,
                    corpus_level=False):
    result = dict()
    N=10
    if corpus_level:
        for i in range(4,5):
            bleu = BLEU(tokenize="spm", max_ngram_order=i)
            tmp = []
            for j in range(math.ceil(len(decoded_preds)/N)):
                tmp.append(bleu.corpus_score(decoded_preds[j*N:(j+1)*N], [decoded_labels[j*N:(j+1)*N]]).score)
            result[f'spbleu_max{i}'] = {"score":mean(tmp)}
    else:
        for i in range(1,5):
            for pred, label in zip(decoded_preds, decoded_labels):
                bleu = BLEU(tokenize="spm", max_ngram_order=i)
                try:
                    result[f'spbleu_max{i}'].append(bleu.corpus_score([pred], [[label]]))
                except:
                    result[f'spbleu_max{i}'] = [bleu.corpus_score([pred], [[label]])]
    return result

def compute_avg_50_spbleu(decoded_preds,
                    decoded_labels,
                    lang,
                    corpus_level=False):
    result = dict()
    N=50
    if corpus_level:
        for i in range(4,5):
            bleu = BLEU(tokenize="spm", max_ngram_order=i)
            tmp = []
            for j in range(math.ceil(len(decoded_preds)/N)):
                tmp.append(bleu.corpus_score(decoded_preds[j*N:(j+1)*N], [decoded_labels[j*N:(j+1)*N]]).score)
            result[f'spbleu_max{i}'] = {"score":mean(tmp)}
    else:
        for i in range(1,5):
            for pred, label in zip(decoded_preds, decoded_labels):
                bleu = BLEU(tokenize="spm", max_ngram_order=i)
                try:
                    result[f'spbleu_max{i}'].append(bleu.corpus_score([pred], [[label]]))
                except:
                    result[f'spbleu_max{i}'] = [bleu.corpus_score([pred], [[label]])]
    return result

def compute_avg_100_spbleu(decoded_preds,
                    decoded_labels,
                    lang,
                    corpus_level=False):
    result = dict()
    N=100
    if corpus_level:
        for i in range(4,5):
            bleu = BLEU(tokenize="spm", max_ngram_order=i)
            tmp = []
            for j in range(math.ceil(len(decoded_preds)/N)):
                tmp.append(bleu.corpus_score(decoded_preds[j*N:(j+1)*N], [decoded_labels[j*N:(j+1)*N]]).score)
            result[f'spbleu_max{i}'] = {"score":mean(tmp)}
    else:
        for i in range(1,5):
            for pred, label in zip(decoded_preds, decoded_labels):
                bleu = BLEU(tokenize="spm", max_ngram_order=i)
                try:
                    result[f'spbleu_max{i}'].append(bleu.corpus_score([pred], [[label]]))
                except:
                    result[f'spbleu_max{i}'] = [bleu.corpus_score([pred], [[label]])]
    return result

def compute_chrf(decoded_preds,
                  decoded_labels,
                  lang,
                  corpus_level=False):
    result = dict()
    if corpus_level:
        for i in range(1, 7):
            chrf = CHRF(char_order=i)
            result[f'spbleu_max{i}'] = chrf.corpus_score(decoded_preds, [decoded_labels])
    else:
        for i in range(1, 5):
            for pred, label in zip(decoded_preds, decoded_labels):
                chrf = CHRF(char_order=i)
                try:
                    result[f'chrf_max{i}'].append(chrf.corpus_score([pred], [[label]]))
                except:
                    result[f'chrf_max{i}'] = [chrf.corpus_score([pred], [[label]])]
    return result

def compute_bertscore(decoded_preds,
                      decoded_labels,
                      lang,
                      corpus_level=False):
    result = dict()
    if corpus_level:
        result['bertscore'] = score(decoded_preds, decoded_labels, model_type="microsoft/deberta-xlarge-mnli", lang=lang, verbose=True, batch_size=128, device="cuda:0")
    else:
        for pred, label in zip(decoded_preds, decoded_labels):
            try:
                result['bertscore'].append(score([pred], [label], model_type="microsoft/deberta-xlarge-mnli", lang=lang, verbose=True, batch_size=32))
            except:
                result[f'bertscore'] = [score([pred], [label], model_type="microsoft/deberta-xlarge-mnli", lang=lang, verbose=True, batch_size=32)]
    return result
