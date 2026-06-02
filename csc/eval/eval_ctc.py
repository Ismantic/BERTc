"""BERTc-CTC 评估框架(对标 pycorrector / MacBERT-CSC)。

输入:
  - gold:    (source, target, type) triples
  - pred:    模型输出的 corrected text

指标:
  - Detection (检测错误位置)P/R/F1
  - Correction (改对)P/R/F1
  - Sentence-level Accuracy(整句完全对)
  - Char-level Accuracy

用法:
  from eval_ctc import load_dataset, evaluate, MockModel
  data = load_dataset('sighan15/test')
  preds = [model.correct(s) for s, t, ty in data]
  results = evaluate(data, preds)
  print(results)
"""
import os, glob, json
from collections import defaultdict


DATA = "/home/tfbao/Shiyu/data/bertc_ctc/data"


def load_dataset(name):
    """统一 loader: 返回 list[(source, target, type)]
    type: 'positive' = 无需纠错 | 'negative' = 需纠错
    """
    if name.startswith('sighan'):
        # sighan13/train, sighan15/test 格式: src\ttgt(无 type)
        path = f"{DATA}/{name}.txt"
        data = []
        with open(path) as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) != 2: continue
                src, tgt = parts
                typ = 'positive' if src == tgt else 'negative'
                data.append((src, tgt, typ))
        return data
    else:
        # shibing624/cscd_ns 格式: src\ttgt\ttype
        path = f"{DATA}/shibing624_csc/{name}.tsv"
        data = []
        with open(path) as f:
            next(f)  # header
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 3: continue
                data.append((parts[0], parts[1], parts[2]))
        return data


def char_diff(src, tgt):
    """两等长 char-level diff -> [(pos, src_char, tgt_char)]"""
    if len(src) != len(tgt):
        return None  # 等长才能 char-level 比较
    return [(i, s, t) for i, (s, t) in enumerate(zip(src, tgt)) if s != t]


def evaluate(data, preds):
    """主评估。
    data: list[(source, target, type)]
    preds: list[str] (模型纠正后的句子)
    返回 dict 含:
      sent_acc:         整句完全对的比例
      char_acc:         char-level 正确比例
      detection_*:      错误位置检测 P/R/F1
      correction_*:     错误纠正(检测 + 改对)P/R/F1
      n / n_pos / n_neg
    """
    assert len(data) == len(preds), f"len mismatch: data={len(data)} preds={len(preds)}"

    # sentence-level
    sent_correct = sum(1 for (s, t, _), p in zip(data, preds) if p == t)

    # 等长样本(char-level 评估前提)
    eq_len = [(s, t, p) for (s, t, _), p in zip(data, preds) if len(s) == len(t) == len(p)]
    skipped = len(data) - len(eq_len)

    char_total = char_correct = 0
    # detection: 找出"哪些 char 错"
    tp_det = fp_det = fn_det = 0
    # correction: detection + 改对
    tp_cor = fp_cor = fn_cor = 0

    for s, t, p in eq_len:
        for i in range(len(s)):
            char_total += 1
            if p[i] == t[i]:
                char_correct += 1

            # detection 视角
            gold_is_err = (s[i] != t[i])
            pred_is_err = (s[i] != p[i])
            if gold_is_err and pred_is_err:
                tp_det += 1
                if p[i] == t[i]:  # 正确改了
                    tp_cor += 1
                else:  # 改了但改错
                    fp_cor += 1; fn_cor += 1  # 既算错的正例也算漏的真例
            elif gold_is_err and not pred_is_err:
                fn_det += 1; fn_cor += 1
            elif not gold_is_err and pred_is_err:
                fp_det += 1; fp_cor += 1

    def prf(tp, fp, fn):
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 2 * p * r / max(1e-9, p + r)
        return p, r, f1

    det_p, det_r, det_f1 = prf(tp_det, fp_det, fn_det)
    cor_p, cor_r, cor_f1 = prf(tp_cor, fp_cor, fn_cor)

    n_pos = sum(1 for _, _, ty in data if ty == 'positive')
    return {
        'n':              len(data),
        'n_pos':          n_pos,
        'n_neg':          len(data) - n_pos,
        'n_eqlen_sample': len(eq_len),
        'n_skip_lenmix':  skipped,
        'sent_acc':       sent_correct / len(data),
        'char_acc':       char_correct / max(1, char_total),
        'detection_P':    det_p,
        'detection_R':    det_r,
        'detection_F1':   det_f1,
        'correction_P':   cor_p,
        'correction_R':   cor_r,
        'correction_F1':  cor_f1,
    }


# === baseline 模型接口(给后续模型用)===

class IdentityBaseline:
    """完全不改,sent_acc 应等于 n_pos / n,F1=0"""
    def correct(self, text):
        return text


class MockOracle:
    """假设知道 gold(只用来 sanity check 评估代码)"""
    def __init__(self, gold_pairs):
        self.gold_map = {s: t for s, t, _ in gold_pairs}
    def correct(self, text):
        return self.gold_map.get(text, text)


def print_results(name, results):
    print(f"\n=== {name} (n={results['n']}, pos={results['n_pos']}, neg={results['n_neg']}) ===")
    print(f"  Sent Acc:       {results['sent_acc']:.4f}")
    print(f"  Char Acc:       {results['char_acc']:.4f}")
    print(f"  Detection F1:   {results['detection_F1']:.4f}  (P={results['detection_P']:.4f} R={results['detection_R']:.4f})")
    print(f"  Correction F1:  {results['correction_F1']:.4f}  (P={results['correction_P']:.4f} R={results['correction_R']:.4f})")
    if results['n_skip_lenmix']:
        print(f"  (skipped {results['n_skip_lenmix']} len-mismatch)")


if __name__ == "__main__":
    # 自检:在 sighan15/test 上跑 Identity baseline + Oracle
    data = load_dataset('sighan15/test')
    print(f"Loaded sighan15/test: {len(data)} samples")

    # Identity baseline(不改)— 应该 sent_acc ≈ n_pos/n
    ident = IdentityBaseline()
    preds = [ident.correct(s) for s, _, _ in data]
    r_id = evaluate(data, preds)
    print_results("Identity (baseline 不改)", r_id)

    # Oracle(完美 — sanity check)
    oracle = MockOracle(data)
    preds_oracle = [oracle.correct(s) for s, _, _ in data]
    r_oracle = evaluate(data, preds_oracle)
    print_results("Oracle (sanity check)", r_oracle)
