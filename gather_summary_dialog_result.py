import glob
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List
from tqdm import tqdm
import numpy as np
import pandas as pd
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
    # ipdb.set_trace()
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
        for special_token in ['<doc_bos>', '<pat_bos>', '<doctor>', '<patient>', '<act>', 
                              '<turn>', '<stage>', '<info>', '<summary>', '<next>']:
            jieba.add_word(special_token, freq=1000000000000000)

    return "#".join(jieba.cut(text.replace(" ", ""))) \
        .replace("<#act#>", "<act>") \
        .replace("<#doc#_#bos#>", "<doc_bos>") \
        .split("#")


def convert_result(data, summary=False):
    formatted_data = []
    w_cot_file = '/home/hy/gyy/psy/Baichuan2/fine-tune/data/w_cot_action/psy_w_cot_test.json'
    wo_cot_file = '/home/hy/gyy/psy/Baichuan2/fine-tune/data/psy_wo_cot_test.json'
    if not summary:
        with open(w_cot_file, encoding="utf8") as f:
            tmp_file = json.load(f)
            labels = [l['conversations'][-1]['action'] for l in tmp_file]
            w_cot_targets = [l['conversations'][-1]['value'].replace('</s>\n', '') for l in tmp_file]
        with open(wo_cot_file, encoding="utf8") as f:
            tmp_file = json.load(f)
            wo_cot_targets = [l['conversations'][-1]['value'] for l in tmp_file]
    for i, d in enumerate(data['result']):
        if summary:
            formatted_data.append([
                "<summary> " + " ".join([_ for _ in d['prediction']]),
                "<summary> " + " ".join([_ for _ in d['reference']]),
            ])
        else:
            if '<next>' in d['generated_text']:
                act = "<next> " + " ".join([_ for _ in re.search(r'<next>(.*?)<', d['generated_text']).group(1)]) + " "
                act_ref = "<next> " + " ".join([_ for _ in labels[i]]) + " "
            elif '下一轮对话的话题' in d['generated_text']:
                act = "<next> " + " ".join([_ for _ in re.search(r'下一轮对话的话题：(.*?)\n', d['generated_text']).group(1)]) + " "
                act_ref = "<next> " + " ".join([_ for _ in labels[i]]) + " "
            elif '<next>' in d['prompt'] or '下一轮对话的话题' in d['prompt']:
                act = act_ref = "<next> " + " ".join([_ for _ in labels[i]]) + " "
            else:
                act_ref = act = '<next> '
            formatted_data.append([
                act + "<doctor> " + " ".join([_ for _ in d['prediction']]),
                act_ref + "<doctor> " + " ".join([_ for _ in d['reference']]),
            ])
    print(formatted_data[0])
    print(formatted_data[-1])
    return formatted_data


def statistical(file: str, remove_prompt=True, convert=False, summary=False, add_token=True):

    if remove_prompt:
        print(f"========= {file} | remove prompt =========")
    else:
        print(f"========= {file} =========")

    with open(file, encoding="utf8") as f:
        labels_and_predictions = json.load(f)
        if convert:
            labels_and_predictions = convert_result(labels_and_predictions, summary=summary)

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
        if not add_token:
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

    # try:
    #     bert_score = compute_bert_score(all_preds, all_labels)[-1]
    #     bert_score = round(bert_score, 4)
    # except:
    bert_score = 0
    
    # bert_score = compute_bert_score([''.join(i.split(" ")) for i in all_preds], 
    #                                 [''.join(i.split(" ")) for i in all_labels])
    # print("bert-score (combined)", bert_score)
    # bert_score = compute_bert_score([''.join(i.split(" ")).split('<doc_bos>')[-1] for i in all_preds], 
    #                                 [''.join(i.split(" ")).split('<doc_bos>')[-1] for i in all_labels])
    # print("bert-score (combined, doc only)", bert_score)

    bleu_scores = round([x / cnt_samples for x in bleu_scores][1], 4)
    nist_scores = [x / cnt_samples for x in nist_scores]
    meteor = round(meteor / cnt_samples, 4)
    rouge_1f = rouge_1f / cnt_samples
    rouge_1p = rouge_1p / cnt_samples
    rouge_1r = rouge_1r / cnt_samples
    rouge_2f = rouge_2f / cnt_samples
    rouge_2p = rouge_2p / cnt_samples
    rouge_2r = rouge_2r / cnt_samples
    rouge_lf = round(rouge_lf / cnt_samples, 4)
    rouge_lp = rouge_lp / cnt_samples
    rouge_lr = rouge_lr / cnt_samples

    dist_score = round(dist[1], 4)
    
    print(f"\n======= {file} | statistical results =======")
    print("bleu", bleu_scores)
    # print("nist", nist_scores)
    print("meteor", meteor)
    # print("rouge-1f,p,r", rouge_1f, rouge_1p, rouge_1r)
    # print("rouge-2f,p,r", rouge_2f, rouge_2p, rouge_2r)
    print("rouge-l", rouge_lf)
    # print("entropy", entropy)
    print("dist", dist_score)
    print("bert-score", bert_score)

    metric = {
        'model': Path(file).stem,
        # 'add_token': add_token,
        'BLEU-2': bleu_scores,
        'ROUGE-L': rouge_lf,
        'METEOR': meteor,
        'DIST-2': dist_score,
        # 'BERT-SCORE': bert_score
    }
    return metric


def action_statistical(file: str, convert: bool):
    from sklearn.metrics import classification_report

    # _labels = {'共情安慰', '兴趣', '其它', '情绪', '睡眠', '社会功能',
    #            '筛查', '精神状态', '自杀倾向', '躯体症状', '食欲'}
    _labels = {'共情安慰', '核心', '其它', '行为', '筛查','自杀倾向'}
    labels = ['共情安慰', '兴趣', '其它', '情绪', '睡眠', '社会功能',
               '筛查', '精神状态', '自杀倾向', '躯体症状', '食欲']

    with open(file, encoding="utf8") as f:
        samples = json.load(f)
        if convert:
            samples = convert_result(samples)

    acc = 0
    pred_actions = []
    label_actions = []

    for pred, label in samples:
        try:
            ptr = "<act>(.*?)<" if '<act>' in label else "<next>(.*?)<"
            pred_action = ''.join(re.search(ptr, pred).group(1).split())
            label_action = ''.join(re.search(ptr, label).group(1).split())
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
    parser.add_argument('-c', '--convert', action="store_true", default=False)
    parser.add_argument('-sum', '--summary', action="store_true", default=False)
    args = parser.parse_args()

    all_results = []

    # for t in [True, False]:
    for file in sorted(glob.glob(f"{args.src_dir}/*.json")):
        metric = statistical(file, remove_prompt=True, convert=args.convert, 
                             add_token=True,
                             summary=args.summary)
        action_statistical(file, convert=args.convert)
        all_results.append(metric)
    
    print(all_results)
    df = pd.DataFrame(all_results)
    df.to_csv(f"{args.src_dir}/all_results.csv", index=False)


if __name__ == '__main__':
    main()
