"""BERTc 数据源注册表 + 路径解析。

所有下载/加工脚本共用这一份配置,避免像 pretokenize_modern.py 那样
把语料路径和用量散落在各处(其默认值甚至跟实跑值对不上)。

落地根由 BERTC_DATA_ROOT 控制,默认指向现有语料位置,因此现有 250G
语料**一行都不用重下**。全新机器上换个路径即可从零拉取。

每个源的 n_parts 默认值 = v4-Large(HF 上的 Ismantic/BERTc-315M)实跑用量,
读自 pretrain/modern_bertc/data3/pretok_v3.log,不是脚本默认值。
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 语料落地根。默认 = 现有位置,零重下。
DATA_ROOT = Path(os.environ.get("BERTC_DATA_ROOT", "/home/tfbao/a6000"))

# 加工产物落地根(documents.txt / jsonl / pkl)。默认同上。
DERIVED_ROOT = Path(os.environ.get("BERTC_DERIVED_ROOT", str(DATA_ROOT / "derived")))

# HF 镜像。置空则走官方源。
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")


@dataclass
class Source:
    """一个可下载的数据源。

    part_glob 相对于 dir();n_parts=None 表示全量。
    legacy_dirs 是 v4-Large 实跑时的历史位置,resolve() 会优先命中已存在的那个,
    这样现有数据直接复用,而全新环境走 DATA_ROOT/subdir 的标准布局。
    """
    name: str
    kind: str                      # "hf" | "github"
    repo_id: str
    subdir: str                    # 相对 DATA_ROOT
    part_glob: str                 # 相对 dir(),用于枚举/截断 part
    n_parts: int | None = None     # None = 全量;默认值 = v4-Large 实跑用量
    allow_patterns: list[str] = field(default_factory=list)
    legacy_dirs: list[str] = field(default_factory=list)
    note: str = ""

    def dir(self) -> Path:
        """解析出实际目录:优先已存在的历史位置,否则标准位置。"""
        for legacy in self.legacy_dirs:
            p = Path(legacy) if os.path.isabs(legacy) else DATA_ROOT / legacy
            if p.exists():
                return p
        return DATA_ROOT / self.subdir

    def files(self, n: int | None = -1) -> list[Path]:
        """按 sorted glob 取前 n 个 part。n=-1 用注册的默认值,None 取全部。"""
        if n == -1:
            n = self.n_parts
        got = sorted(self.dir().glob(self.part_glob))
        return got if n is None else got[:n]


# ---------------------------------------------------------------- 预训练语料
# v4-Large 实跑(data3/train_v3.pt,17.65B token,zh:en = 6:4):
#   wiki_cn 3 / PeopleDaily 1 / SkyPile 21 / CCI3-HQ 5 / FineWeb-Edu 500
#   CnnDM 1 / wiki_en 25

PRETRAIN_SOURCES = {
    "skypile": Source(
        name="skypile", kind="hf",
        repo_id="Skywork/SkyPile-150B",
        subdir="SkyPile", part_glob="*.parquet",
        n_parts=21,
        allow_patterns=["*.parquet"],
        note="中文网页。盘上现有 42 个 parquet,v4-Large 用了 0-20。",
    ),
    "cci3": Source(
        name="cci3", kind="hf",
        repo_id="BAAI/CCI3-HQ",
        subdir="CCI3-HQ", part_glob="data/*.jsonl",
        n_parts=5,
        allow_patterns=["data/*.jsonl"],
        legacy_dirs=["Summer-data/CCI3-HQ"],
        note="中文高质量语料。v4-Large 用了全部 5 个 jsonl(4.9G)。"
             "HF 上全量远不止 5 个,扩量时调 --n-parts。",
    ),
    "fineweb_edu_zh": Source(
        name="fineweb_edu_zh", kind="hf",
        repo_id="opencsg/Fineweb-Edu-Chinese-V2.2",
        subdir="Chinese-FineWeb-Edu-V2.2", part_glob="4_5/*.parquet",
        n_parts=500,
        allow_patterns=["4_5/*.parquet"],
        legacy_dirs=["Summer-data/Chinese-FineWeb-Edu-V2.2"],
        note="中文教育向。只用打分 4_5 的子集;盘上 9745 个 parquet,v4-Large 用了 0-499。",
    ),
    "finewiki_zh": Source(
        name="finewiki_zh", kind="hf",
        repo_id="HuggingFaceFW/finewiki",
        subdir="finewiki", part_glob="data/zhwiki/*.parquet",
        n_parts=None,
        allow_patterns=["data/zhwiki/*.parquet"],
        note="中文维基。5 parquet / 5.53GB。取代 v4-Large 用的 2023-11 json dump"
             "(Wikipedia_cn_json_files,3 files 501MB)。",
    ),
    "finewiki_en": Source(
        name="finewiki_en", kind="hf",
        repo_id="HuggingFaceFW/finewiki",
        subdir="finewiki", part_glob="data/enwiki/*.parquet",
        n_parts=None,
        allow_patterns=["data/enwiki/*.parquet"],
        note="英文维基。15 parquet / 37.72GB(全量)。取代 v4-Large 用的 25 个旧 json file。",
    ),
    "people_daily": Source(
        name="people_daily", kind="hf",
        repo_id="Papersnake/people_daily_news",
        subdir="people_daily_news", part_glob="*.jsonl.gz",
        n_parts=None,
        legacy_dirs=["/home/tfbao/Shiyu/Data/data/people_daily_news"],
        note="人民日报 1946-2025 全文。需 process.py 加工成 documents.txt 才能喂 pretokenize。",
    ),
    "cnn_dailymail": Source(
        name="cnn_dailymail", kind="hf",
        repo_id="abisee/cnn_dailymail",
        subdir="cnn_dailymail", part_glob="**/*.parquet",
        n_parts=None,
        legacy_dirs=["/home/tfbao/Shiyu/Data/data/cnn_dailymail"],
        note="英文新闻。同样需 process.py 加工。",
    ),
}

# ---------------------------------------------------------------- 下游任务数据

FINETUNE_SOURCES = {
    "pd1998": Source(
        name="pd1998", kind="github",
        repo_id="chenhui-bupt/PeopleDaily1998",
        subdir="PeopleDaily1998", part_glob="199801.zip",
        n_parts=None,
        note="PD-1998 PFR 标注语料(仅 1998 年 1 月,20.3MB zip)。"
             "process_cws.py 解析成 cws/pos/ner 三份 jsonl。",
    ),
}

# CSC 训练对的原始源。当初是手工逐个下的,dl 日志显示不少 repo 报 401/404,
# 落到盘上的这批没有留下完整的 repo_id 记录 —— 这里只登记能确认的,
# 其余保持现状(process_csc.py 直接扫 CSC_RAW_DIR,不依赖本表)。
CSC_RAW_DIR = Path(os.environ.get(
    "BERTC_CSC_RAW", str(REPO_ROOT / "csc" / "data" / "raw")))

CSC_SOURCES = {
    "wang271k": Source(
        name="wang271k", kind="hf",
        repo_id="shibing624/CSC",
        subdir="csc/wang271k_csc", part_glob="*.json",
        n_parts=None,
        legacy_dirs=[str(REPO_ROOT / "csc" / "data" / "raw" / "wang271k_csc")],
        note="Wang271K + SIGHAN,MacBERT4CSC 的标准训练集。",
    ),
}

ALL_SOURCES = {**PRETRAIN_SOURCES, **FINETUNE_SOURCES, **CSC_SOURCES}


# ---------------------------------------------------------------- 加工产物

DERIVED = {
    # pretokenize 直接读的两个 documents.txt
    "people_daily_docs": "PeopleDaily.documents.txt",
    "cnn_dailymail_docs": "CnnDailyMail.documents.txt",
}

# v4-Large 实跑时这两个文件的位置(Shiyu/Data 产出),resolve 时优先复用
DERIVED_LEGACY = {
    "people_daily_docs": "/home/tfbao/Shiyu/Data/data/PeopleDaily.documents.txt",
    "cnn_dailymail_docs": "/home/tfbao/Shiyu/Data/data/CnnDailyMail.documents.txt",
}


def derived_path(key: str) -> Path:
    """加工产物路径:已存在的历史位置优先,否则 DERIVED_ROOT 下的标准位置。"""
    legacy = DERIVED_LEGACY.get(key)
    if legacy and Path(legacy).exists():
        return Path(legacy)
    return DERIVED_ROOT / DERIVED[key]


def describe() -> str:
    """打印当前解析结果,用于确认路径是否如预期。"""
    lines = [f"DATA_ROOT    = {DATA_ROOT}",
             f"DERIVED_ROOT = {DERIVED_ROOT}",
             f"HF_ENDPOINT  = {HF_ENDPOINT or '(官方源)'}",
             ""]
    for name, src in ALL_SOURCES.items():
        d = src.dir()
        have = len(list(d.glob(src.part_glob))) if d.exists() else 0
        want = "全部" if src.n_parts is None else src.n_parts
        mark = "✓" if have else "✗"
        lines.append(f"  {mark} {name:<16} {have:>5} 个 part(需 {want})  {d}")
    lines.append("")
    for key in DERIVED:
        p = derived_path(key)
        lines.append(f"  {'✓' if p.exists() else '✗'} {key:<20} {p}")
    lines.append("")
    lines.append(f"  CSC raw: {CSC_RAW_DIR}"
                 f"  ({'存在' if CSC_RAW_DIR.exists() else '缺失'})")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
