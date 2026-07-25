"""CWS / POS / NER 的标签表,以及 PD-1998 标注到 LTP 标签体系的映射。

只有 prepare/ 用 —— 标签体系是对**文本**的定义,src/ 只认 id。
标签名会随预编码数据一起写进文件(cws_vocab / pos_vocab / ner_vocab),
src/evaluate.py 靠这些名字还原出 span,所以顺序不能改。

"""

# ---------------------------------------------------------------- CWS

CWS_TAGS = ["B", "I", "E", "S"]
CWS2ID = {t: i for i, t in enumerate(CWS_TAGS)}


def words_to_cws_bies(words):
    """词列表 → (字列表, BIES 标签 id 列表)。"""
    chars, tags = [], []
    for w in words:
        if not w:
            continue
        if len(w) == 1:
            chars.append(w)
            tags.append(CWS2ID["S"])
        else:
            chars.append(w[0])
            tags.append(CWS2ID["B"])
            for c in w[1:-1]:
                chars.append(c)
                tags.append(CWS2ID["I"])
            chars.append(w[-1])
            tags.append(CWS2ID["E"])
    return chars, tags


# ---------------------------------------------------------------- POS

# PD-1998 的 43 个词性 → LTP base1 的 27 个。
# 映射依据是实测:抽 PD 里每个标签的高频词,单独喂 LTP 看它实际归到哪类,
# 而不是照字面对应 —— 兼类词(ad/an/vn/vd)按字面会对错。
PD2LTP_POS = {
    "a": "a", "Ag": "a",
    "b": "b", "Bg": "b",
    "c": "c",
    "d": "d",
    "e": "e", "Yg": "e",
    "h": "h",
    "i": "i", "l": "i",
    "j": "j",
    "k": "k",
    "m": "m", "Mg": "m",
    "n": "n", "Ng": "n",
    "o": "o",
    "p": "p",
    "q": "q",
    "r": "r", "Rg": "r",
    "u": "u",
    "v": "v", "Vg": "v",
    "z": "z",
    # 兼类词:LTP 按词义判,实测后修正
    "ad": "a",   # 积极 / 全面 / 努力 → LTP 标 a,不是 d
    "an": "a",   # 困难 / 稳定 → LTP 标 a,不是 n
    "vn": "v",   # 工作 / 建设 / 发展 → LTP 标 v,不是 n
    "vd": "v",   # 持续 / 免费 → LTP 标 v,不是 d
    "y": "u",    # 了 / 呢 / 吗 → LTP 标 u 助词,不是 e 叹词
    # 重映射
    "nr": "nh",  # 人名
    "ns": "ns",
    "nt": "ni",  # 机构。注意 LTP 的 nt 是时间,别搞反
    "nz": "nz",
    "nx": "x",   # 外文字符
    "f": "nd",   # 方位
    "s": "nl",   # 处所
    "t": "nt", "Tg": "nt",   # 时间 → LTP nt
    "w": "wp",   # 标点
}

POS_TAGS = [
    "a", "b", "c", "d", "e", "h", "i", "j", "k", "m",
    "n", "nd", "nh", "ni", "nl", "ns", "nt", "nz", "o", "p",
    "q", "r", "u", "v", "wp", "x", "z",
]
POS2ID = {t: i for i, t in enumerate(POS_TAGS)}


def map_pd_pos(pd_tag: str) -> str:
    """PD 词性 → LTP 词性,认不出的落到 x。"""
    return PD2LTP_POS.get(pd_tag, "x")


# ---------------------------------------------------------------- NER

NER_TAGS = [
    "O",
    "B-Nh", "I-Nh", "E-Nh", "S-Nh",     # 人名
    "B-Ns", "I-Ns", "E-Ns", "S-Ns",     # 地名
    "B-Ni", "I-Ni", "E-Ni", "S-Ni",     # 机构
]
NER2ID = {t: i for i, t in enumerate(NER_TAGS)}

# PD 的实体类型 → LTP 类型。MISC / I / L 一律丢弃。
PD2LTP_NER = {"PER": "Nh", "LOC": "Ns", "ORG": "Ni"}


def entity_to_bies(ner_tags: list, start: int, end: int, ent_type: str) -> None:
    """把一个实体区间就地写成 BIES 标签。类型认不出或区间越界就不写。"""
    ent_type = PD2LTP_NER.get(ent_type, ent_type)
    if ent_type not in ("Nh", "Ns", "Ni"):
        return
    n = len(ner_tags)
    end = min(end, n)
    if start >= n or end - start <= 0:
        return
    if end - start == 1:
        ner_tags[start] = NER2ID[f"S-{ent_type}"]
    else:
        ner_tags[start] = NER2ID[f"B-{ent_type}"]
        for j in range(start + 1, end - 1):
            ner_tags[j] = NER2ID[f"I-{ent_type}"]
        ner_tags[end - 1] = NER2ID[f"E-{ent_type}"]
