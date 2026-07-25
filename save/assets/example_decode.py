"""BERTc-MT:分词 + 词性 + 命名实体。"""
from mt_model import BERTcForMT

model = BERTcForMT.from_pretrained(".")
for r in model.predict([
    "中国科学院计算技术研究所在北京",
    "李雷和韩梅梅去上海参加了会议",
]):
    print(r["text"])
    print("  分词:", " / ".join(r["words"]))
    print("  词性:", " ".join(f"{w}/{p}" for w, p in zip(r["words"], r["pos"])))
    print("  实体:", r["ner"])
