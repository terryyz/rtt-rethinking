import os
import argparse
import glob
import pickle
from models import *
from local_datasets import *
from metrics.run_metric import *
from utils.constants import *


def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model-name', type=str, default='mbart50-m2m',
                        choices=["mbart50-m2m", "m2m-100-base", "m2m-100-large", "m2m-124", "fair-wmt20","GPT-3", "Helsinki-NLP", "google_drive"],
                        help='language model name')
    parser.add_argument('--dataset', type=str, default='flores_101',
                        choices=["wmt14", "wmt15", "wmt16", "wmt17", "wmt18", "wmt19", "wmt20-news", "wmt20-bio","flores_101"],
                        help='dataset name')
    parser.add_argument('--start', type=float)
    parser.add_argument('--end', type=float)
    parser.add_argument('--skip-src', type=str)
    parser.add_argument('--skip-tgt', type=str)
    parser.add_argument('--corpus-level', type=bool, default=True,
                        help='corpus-level or sample-level')
    parser.add_argument('--cuda', type=str, default='1',
                        help='cuda index')
    return parser

class Features(object):

    def __init__(self, args: dict, src_lang, tgt_lang):
        self.cuda = args.cuda
        self.model_name = args.model_name
        self.dataset_name = args.dataset
        self.src_lang, self.tgt_lang = src_lang, tgt_lang
        self._corpus_level = args.corpus_level
        if self.model_name == "fair-wmt20":
            torch.hub.set_dir('tmp_model')
            print(torch.hub.list('pytorch/fairseq'))
            model = dict()
            model['en-ta'] = torch.hub.load('pytorch/fairseq', 'transformer.wmt20.en-ta').to(f'cuda:{self.cuda}')
            model['ta-en'] = torch.hub.load('pytorch/fairseq', 'transformer.wmt20.ta-en').to(f'cuda:{self.cuda}')
            self.model = model
        else:
            self.model = load_model(self.model_name, self.cuda)


    def sub_process(self, _test=False, _corpus_level=False):
        split = "devtest" if _test else "dev"
        split_name = "test" if _test else "train"
        src_ref_train, tgt_ref_train = read_data(self.dataset_name, self.src_lang, self.tgt_lang, split)
        dir_path = f"local_{self.dataset_name}/{self.src_lang}_{self.tgt_lang}/{self.model_name}"
        # source -> target
        try:
            med_tgt_train = pickle.load(open(f'{dir_path}/{split_name}_trans_from_tgt.pkl',"rb"))
        except:
            print(f"Can't find {dir_path}/{split_name}_trans_from_tgt.pkl")
            med_tgt_train = model_translate(self.model, src_ref_train, self.src_lang, self.tgt_lang)

        # target -> source
        try:
            med_src_train = pickle.load(open(f'{dir_path}/{split_name}_trans_from_src.pkl',"rb"))
        except:
            print(f"Can't find {dir_path}/{split_name}_trans_from_src.pkl")
            med_src_train = model_translate(self.model, tgt_ref_train, self.tgt_lang, self.src_lang)

        direct_score = Scores(src_ref_train, med_src_train, tgt_ref_train, med_tgt_train,
                              self.src_lang, self.tgt_lang,
                              self.dataset_name, self.model_name, ["spbleu","bleu","chrf","bertscore"],
                              _test=_test, _corpus_level=self._corpus_level)
        direct_score.store_translation_results()
        direct_score.store_score_results()

        # source -> target -> source
        try:
            src_pred_train = pickle.load(open(f'{dir_path}/{split_name}_self_src.pkl',"rb"))
        except:
            print(f"Can't find {dir_path}/{split_name}_self_src.pkl")
            src_pred_train = model_translate(self.model,
                                         med_tgt_train, self.tgt_lang,
                                         self.src_lang)

        #target -> source -> target
        try:
            tgt_pred_train = pickle.load(open(f'{dir_path}/{split_name}_self_tgt.pkl',"rb"))
        except:
            print(f"Can't find {dir_path}/{split_name}_self_tgt.pkl")
            tgt_pred_train = model_translate(self.model,
                                         med_src_train, self.src_lang,
                                         self.tgt_lang)

        reverse_score = Scores(src_ref_train, src_pred_train, tgt_ref_train, tgt_pred_train,
                               self.src_lang, self.tgt_lang,
                               self.dataset_name, self.model_name, ["spbleu","bleu","chrf","bertscore"],
                               _test=_test, _self=True, _corpus_level=self._corpus_level)
        reverse_score.store_translation_results()
        reverse_score.store_score_results()

    def run(self):
        # try:
        # print("Start Preparing Training Data!")
        # self.sub_process()
        # except:
        #     passd
        print("Start Preparing Test Data!")
        self.sub_process(_test=True)

if __name__ == "__main__":
    args = get_args().parse_args()
    flg = False
    model_lang_dict = {
        "m2m-100-large": M2M100_LANGS,
        "m2m-100-base": M2M100_LANGS,
        "mbart50-m2m": MBART50_LANGS,
        "Helsinki-NLP": HELSINKI_LANGS
    }
    m2m_langs = [lang.split("_")[0] for lang in model_lang_dict[args.model_name]]
    if args.dataset == "flores_101":
        lang_pairs = [(x, y) for x, y in itertools.combinations(REGION_1, r=2) if x != y] + list(itertools.product(REGION_1, REGION_2)) + list(itertools.product(REGION_2, REGION_1)) + [(x, y) for x, y in itertools.combinations(REGION_2, r=2) if x != y]
        folders=glob.glob(f"local_flores_101/*")
        for src_lang, tgt_lang in lang_pairs[int(args.start * len(lang_pairs)):int(args.end * len(lang_pairs))]:
            try:
                files = glob.glob(f"local_flores_101/{src_lang}_{tgt_lang}/{args.model_name}/*")
                if args.skip_src == 'none' and args.skip_tgt == 'none':
                    flg = True
                if flg and src_lang in m2m_langs and tgt_lang in m2m_langs and len(files) > 0:# and len(files) < 44:# and f"local_flores_101/{src_lang}_{tgt_lang}" not in folders and f"local_flores_101/{tgt_lang}_{src_lang}" not in folders:
                    print(src_lang, tgt_lang)
                    print("======================")
                    features = Features(args, src_lang, tgt_lang)
                    features.run()
                    print()
                if src_lang == args.skip_src and tgt_lang == args.skip_tgt:
                    flg = True
            except:
                pass
    elif args.dataset == "wmt20-news":
        files = glob.glob("data/wmt20/sources/*")
        lang_pairs = [((file.split("/")[-1].split("-")[1][:2]),(file.split("/")[-1].split("-")[1][2:])) for file in files]
        for src_lang, tgt_lang in lang_pairs[int(args.start * len(lang_pairs)):int(args.end * len(lang_pairs))]:

            files = glob.glob(f"local_wmt20-news/{src_lang}_{tgt_lang}/{args.model_name}/*")
            if args.skip_src == 'none' and args.skip_tgt == 'none':
                flg = True
            print(src_lang, tgt_lang, flg)
            print(len(files))
            if flg and len(files) < 24 and ((f"{src_lang}-{tgt_lang}" in m2m_langs and f"{tgt_lang}-{src_lang}" in m2m_langs) or (src_lang in m2m_langs and tgt_lang in m2m_langs)):
                print(src_lang, tgt_lang)
                print("======================")
                features = Features(args, src_lang, tgt_lang)
                features.run()
                print()
            if src_lang == args.skip_src and tgt_lang == args.skip_tgt:
                flg = True
	
    elif args.dataset == "wmt20-bio":
        files = glob.glob("data/wmt20bio/sources/*")
        lang_pairs = [((file.split("/")[-1].split("_")[0][:2]),(file.split("/")[-1].split("-")[0][3:5])) for file in files]
        print(lang_pairs)
        for src_lang, tgt_lang in lang_pairs[int(args.start * len(lang_pairs)):int(args.end * len(lang_pairs))]:

            files = glob.glob(f"local_wmt20-bio/{src_lang}_{tgt_lang}/{args.model_name}/*")
            if args.skip_src == 'none' and args.skip_tgt == 'none':
                flg = True
            print(src_lang, tgt_lang, flg)
            print(len(files))
            # and len(files) < 20
            if flg  and ((f"{src_lang}-{tgt_lang}" in m2m_langs and f"{tgt_lang}-{src_lang}" in m2m_langs) or (src_lang in m2m_langs and tgt_lang in m2m_langs)):
                print(src_lang, tgt_lang)
                print("======================")
                features = Features(args, src_lang, tgt_lang)
                features.run()
                print()
            if src_lang == args.skip_src and tgt_lang == args.skip_tgt:
                flg = True
	
    elif args.model_name == "fair-wmt20":
        lang_pairs = [('ta','en'),('en','ta')]
        for src_lang, tgt_lang in lang_pairs[int(args.start*len(lang_pairs)):int(args.end*len(lang_pairs))]:
            print(src_lang, tgt_lang)
            print("======================")
            features = Features(args, src_lang, tgt_lang)
            features.run()
            print()
