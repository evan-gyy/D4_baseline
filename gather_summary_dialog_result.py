import glob
import json
import re
import argparse
from collections import defaultdict
from typing import List
from tqdm import tqdm
import numpy as np
from bert_score import score, BERTScorer
import jieba
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.nist_score import sentence_nist
import ipdb

_add_special_token = True


def compute_bert_score(prediction, reference):
    scorer = BERTScorer(model_type="/home/hy/models/bert-base-chinese/", num_layers=8)
    P, R, F1 = scorer.score(cands=prediction, 
                            refs=reference,)
    P, R, F1 = [np.average(s) for s in (P, R, F1)]
    return P, R, F1


def compute_bleu(prediction, reference, n):

    temp_bleu = sentence_bleu(
        [reference], prediction, weights=tuple(1 / n for _ in range(n))
    )
    return temp_bleu


def compute_nist(predict, target, n):

    if len(predict) < n or len(target) < n:
        return 0
    return sentence_nist([target], predict, n)


def cal_entropy(all_sentences: List[List[str]]):

    etp_score = [0.0, 0.0, 0.0, 0.0]
    div_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)]
    for sentence in all_sentences:
        for n in range(4):
            for idx in range(len(sentence) - n):
                ngram = " ".join(sentence[idx: idx + n + 1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            etp_score[n] += -(v + 0.0) / total * (np.log(v + 0.0) - np.log(total))
        div_score[n] = (len(counter[n].values()) + 0.0) / total
    return etp_score, div_score


def cut(text: str):
    
    global _add_special_token
    if _add_special_token:
        _add_special_token = False
        for special_token in ['<doc_bos>', '<pat_bos>', '<act>']:
            jieba.add_word(special_token, freq=1000000000000000)

    return "#".join(jieba.cut(text.replace(" ", ""))) \
        .replace("<#act#>", "<act>") \
        .replace("<#doc#_#bos#>", "<doc_bos>") \
        .split("#")


def statistical(file: str, remove_prompt=True):

    if remove_prompt:
        print(f"========= {file} | remove prompt =========")
    else:
        print(f"========= {file} =========")

    with open(file, encoding="utf8") as f:
        labels_and_predictions = json.load(f)

    bleu_scores = [0, 0, 0, 0]
    nist_scores = [0, 0, 0, 0]
    meteor = 0
    cnt_samples = len(labels_and_predictions)
    rouge_1f = rouge_1p = rouge_1r = rouge_2f = rouge_2p = rouge_2r = rouge_lf = rouge_lp = rouge_lr = 0
    all_labels = []
    all_preds = []
    all_prediction_cut = []

    for prediction, label in tqdm(labels_and_predictions):
        prediction = re.sub(r"\[.*?]", "", prediction)
        label = re.sub(r"\[.*?]", "", label)

        # Combined & Doc-only
        prediction = ''.join(prediction.split(" ")).split('<doc_bos>')[-1]
        label = ''.join(label.split(" ")).split('<doc_bos>')[-1]

        if remove_prompt:
            prediction = re.sub("<.*?pat_bos>", "", prediction)
            label = re.sub("<.*?pat_bos>", "", label)
        cut_prediction = cut(prediction)
        cut_label = cut(label)

        all_prediction_cut.append(cut_prediction)
        all_labels.append(label)
        all_preds.append(prediction)
        # ipdb.set_trace()
        for n in range(4):
            bleu_scores[n] += compute_bleu(cut_prediction, cut_label, n + 1)
            nist_scores[n] += compute_nist(cut_prediction, cut_label, n + 1)
        meteor += meteor_score([" ".join(cut_label)], " ".join(cut_prediction))
        rouge = Rouge()
        try:
            rouge_score = rouge.get_scores(" ".join(cut_prediction), " ".join(cut_label), avg=True)
        except ValueError:
            rouge_score = rouge.get_scores("。", " ".join(cut_label), avg=True)

        rouge_1f += rouge_score["rouge-1"]["f"]
        rouge_1p += rouge_score["rouge-1"]["p"]
        rouge_1r += rouge_score["rouge-1"]["r"]
        rouge_2f += rouge_score["rouge-2"]["f"]
        rouge_2p += rouge_score["rouge-2"]["p"]
        rouge_2r += rouge_score["rouge-2"]["r"]
        rouge_lf += rouge_score["rouge-l"]["f"]
        rouge_lp += rouge_score["rouge-l"]["p"]
        rouge_lr += rouge_score["rouge-l"]["r"]

    entropy, dist = cal_entropy(all_prediction_cut)

    bert_score = compute_bert_score(all_preds, all_labels)
    
    # bert_score = compute_bert_score([''.join(i.split(" ")) for i in all_preds], 
    #                                 [''.join(i.split(" ")) for i in all_labels])
    # print("bert-score (combined)", bert_score)
    # bert_score = compute_bert_score([''.join(i.split(" ")).split('<doc_bos>')[-1] for i in all_preds], 
    #                                 [''.join(i.split(" ")).split('<doc_bos>')[-1] for i in all_labels])
    # print("bert-score (combined, doc only)", bert_score)

    bleu_scores = [x / cnt_samples for x in bleu_scores]
    nist_scores = [x / cnt_samples for x in nist_scores]
    meteor = meteor / cnt_samples
    rouge_1f = rouge_1f / cnt_samples
    rouge_1p = rouge_1p / cnt_samples
    rouge_1r = rouge_1r / cnt_samples
    rouge_2f = rouge_2f / cnt_samples
    rouge_2p = rouge_2p / cnt_samples
    rouge_2r = rouge_2r / cnt_samples
    rouge_lf = rouge_lf / cnt_samples
    rouge_lp = rouge_lp / cnt_samples
    rouge_lr = rouge_lr / cnt_samples
    
    print(f"\n======= {file} | statistical results =======")
    print("bleu", bleu_scores)
    # print("nist", nist_scores)
    print("meteor", meteor)
    # print("rouge-1f,p,r", rouge_1f, rouge_1p, rouge_1r)
    # print("rouge-2f,p,r", rouge_2f, rouge_2p, rouge_2r)
    print("rouge-l: f,p,r", rouge_lf, rouge_lp, rouge_lr)
    # print("entropy", entropy)
    print("dist", dist)
    print("bert-score", bert_score)


def action_statistical(file: str):
    from sklearn.metrics import classification_report

    # _labels = {'共情安慰', '兴趣', '其它', '情绪', '睡眠', '社会功能',
    #            '筛查', '精神状态', '自杀倾向', '躯体症状', '食欲'}
    _labels = {'共情安慰', '核心', '其它', '行为', '筛查','自杀倾向'}
    labels = ['共情安慰', '兴趣', '其它', '情绪', '睡眠', '社会功能',
               '筛查', '精神状态', '自杀倾向', '躯体症状', '食欲']

    with open(file, encoding="utf8") as f:
        samples = json.load(f)

    acc = 0
    pred_actions = []
    label_actions = []

    for pred, label in samples:
        try:
            pred_action = ''.join(re.search("<act>(.*?)<", pred).group(1).split())
            label_action = ''.join(re.search("<act>(.*?)<", label).group(1).split())
            if label_action not in _labels or pred_action not in _labels:
                continue    
            acc += (pred_action == label_action)
            pred_actions.append(pred_action)
            label_actions.append(label_action)
        except AttributeError:
            print(f"No action found in '{pred}' or '{label}'")

    print(f"{acc} / {len(samples)} = {(acc / len(samples)):.3%}")
    # print(classification_report(label_actions, pred_actions, digits=4))


def main():
    parser = argparse.ArgumentParser('Gather Summary Result',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--src_dir', type=str, required=True,
                        help='file directory to preprocess')
    args = parser.parse_args()
    
    for file in sorted(glob.glob(f"{args.src_dir}/*/result.json")):
        statistical(file, remove_prompt=True)
        action_statistical(file)


if __name__ == '__main__':
    main()
