"""LTP-aligned label sets for POS + NER。

POS: PD-1998 43 tag → LTP 27 tag (fixed alphabetic order)
NER: BIO 9 tag → BIES 13 tag (Nh/Ns/Ni × {B,I,E,S} + O, 弃 MISC)
"""

# PD-1998 → LTP-base1 POS mapping
# 设计依据:抽样 PD top-3 word 用 LTP/base1 在 carrier-free 单 word 上 inference,
# 看 LTP 实际归类,以此选 mapping。
PD2LTP_POS = {
    # 直接保留 / 同名
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
    # 兼类词:LTP 实际按词义判,验证后修正
    "ad": "a",  # 积极/全面/努力 → LTP 标 a(不是 d)
    "an": "a",  # 困难/稳定/努力 → LTP 标 a(不是 n)
    "vn": "v",  # 工作/建设/发展 → LTP 标 v(不是 n)
    "vd": "v",  # 持续/免费 → LTP 标 v(不是 d)
    "y": "u",   # 了/呢/吗 → LTP 标 u 助词(不是 e 叹词)
    # 重映射
    "nr": "nh",       # 人名
    "ns": "ns",
    "nt": "ni",       # 机构(注意 LTP nt = 时间)
    "nz": "nz",
    "nx": "x",        # 外文字符(LTP 实际多标 ws,但 PD nx 词次很少,保留 x)
    "f": "nd",        # 方位
    "s": "nl",        # 处所
    "t": "nt", "Tg": "nt",   # 时间 → LTP nt
    "w": "wp",        # 标点
}

# LTP 27-tag fixed vocab(alphabetical,跟 LTP/base1 输出对齐)
LTP_POS_TAGS = [
    'a', 'b', 'c', 'd', 'e', 'h', 'i', 'j', 'k', 'm',
    'n', 'nd', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'o', 'p',
    'q', 'r', 'u', 'v', 'wp', 'x', 'z',
]
LTP_POS2ID = {t: i for i, t in enumerate(LTP_POS_TAGS)}
LTP_ID2POS = {i: t for i, t in enumerate(LTP_POS_TAGS)}


def map_pd_pos(pd_tag):
    """PD POS tag → LTP POS tag; 未知 fallback 'x'。"""
    return PD2LTP_POS.get(pd_tag, 'x')


def map_pd_pos_seq(seq):
    return [map_pd_pos(t) for t in seq]


# NER LTP scheme: BIES + 3 type
LTP_NER_TAGS = [
    "O",
    "B-Nh", "I-Nh", "E-Nh", "S-Nh",
    "B-Ns", "I-Ns", "E-Ns", "S-Ns",
    "B-Ni", "I-Ni", "E-Ni", "S-Ni",
]
LTP_NER2ID = {t: i for i, t in enumerate(LTP_NER_TAGS)}
LTP_ID2NER = {i: t for i, t in enumerate(LTP_NER_TAGS)}

# PD type → LTP type
PD2LTP_NER = {
    "PER": "Nh",
    "LOC": "Ns",
    "ORG": "Ni",
    # MISC, I, L 弃
}


def entity_to_bies(start, end, ent_type, n_chars):
    """给一个 entity (char-level [start, end)) → BIES tag id list per char.
    返回 (tag_ids, ok)。ok=False 表 entity 超界。
    """
    if start >= n_chars: return None
    end = min(end, n_chars)
    length = end - start
    if length <= 0: return None
    ltp_type = PD2LTP_NER.get(ent_type)
    if ltp_type is None: return None  # 弃
    tags = []
    if length == 1:
        tags.append((start, LTP_NER2ID[f"S-{ltp_type}"]))
    else:
        tags.append((start, LTP_NER2ID[f"B-{ltp_type}"]))
        for j in range(start + 1, end - 1):
            tags.append((j, LTP_NER2ID[f"I-{ltp_type}"]))
        tags.append((end - 1, LTP_NER2ID[f"E-{ltp_type}"]))
    return tags


def bies_tags_to_spans(tag_ids):
    """BIES id seq → entity spans [(type, start, end)]"""
    spans = []
    i = 0
    n = len(tag_ids)
    while i < n:
        t = LTP_ID2NER.get(int(tag_ids[i]), "O")
        if t.startswith("S-"):
            spans.append((t[2:], i, i + 1))
            i += 1
        elif t.startswith("B-"):
            et = t[2:]
            j = i + 1
            while j < n:
                tj = LTP_ID2NER.get(int(tag_ids[j]), "O")
                if tj == f"I-{et}":
                    j += 1
                elif tj == f"E-{et}":
                    j += 1
                    spans.append((et, i, j))
                    break
                else:
                    break
            else:
                # 没遇到 E,放弃这个 span(broken)
                pass
            i = j
        else:
            i += 1
    return spans


if __name__ == "__main__":
    # sanity
    print(f"LTP POS vocab ({len(LTP_POS_TAGS)}): {LTP_POS_TAGS}")
    print(f"LTP NER vocab ({len(LTP_NER_TAGS)}): {LTP_NER_TAGS}")
    print()
    # PD coverage
    pd_tags = ['n','v','w','u','d','vn','m','p','a','nr','r','ns','c','q','t',
               'f','j','b','ad','l','i','Ng','nz','s','nt','an','y','Vg','z',
               'k','vd','nx','Tg','Ag','h','e','o','Rg','Mg','Bg','Yg']
    print("PD → LTP coverage:")
    for t in pd_tags:
        print(f"  {t:4s} → {map_pd_pos(t)}")
