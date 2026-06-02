"""直接调用 pycorrector 自家的 MacBertCorrector + 他们的 eval_model_batch.
看他们用自己的 pipeline 跑出来什么数字。"""
import sys
sys.path.insert(0, '/home/tfbao/Shiyu/data/bertc_ctc/pycorrector')

# 用本地下载的 ckpt 而不是再 HF 下载
MODEL = "/home/tfbao/Shiyu/data/bertc_ctc/macbert4csc_model"
TEST  = "/home/tfbao/Shiyu/data/bertc_ctc/pycorrector/pycorrector/data/sighan2015_test.tsv"

# 关掉 verbose 看清最终数字
import io, contextlib

print(">>> Loading pycorrector MacBertCorrector...", flush=True)
from pycorrector.macbert.macbert_corrector import MacBertCorrector
model = MacBertCorrector(model_name_or_path=MODEL)
print(">>> Loaded.", flush=True)

print(">>> Running eval_model_batch...", flush=True)
from pycorrector.utils.evaluate_utils import eval_model_batch

# verbose=False 减少输出
acc, precision, recall, f1 = eval_model_batch(
    model.correct_batch, input_tsv_file=TEST, verbose=False
)
print(f"\n=== FINAL (pycorrector 完全原版) ===")
print(f"Sentence Acc:       {acc:.4f}")
print(f"Sentence Precision: {precision:.4f}")
print(f"Sentence Recall:    {recall:.4f}")
print(f"Sentence F1:        {f1:.4f}")
print(f"\n模型卡报: Sentence F1 = 0.7789")
