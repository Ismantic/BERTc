"""BERTc-CSC:中文拼写纠错。"""
from csc_model import BERTcForCSC

model = BERTcForCSC.from_pretrained(".")
for s in ["我今天很稿兴", "他的身体健康状况很不错,平时喜欢锻练"]:
    print(f"{s}  →  {model.correct(s)}")
