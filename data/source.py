"""BERTc 数据源注册表 + 路径解析。

**这个仓库的目标是:只靠 Hugging Face 和 GitHub 就能从零重训 BERTc。**
所以这里登记的每个源都必须有公开出处,不允许指向本机的既有目录 ——
一旦允许"本地已有就跳过下载",别人克隆下来跑不通,而自己永远发现不了。

下载落在 BERTC_DATA_ROOT(默认仓库内 data/downloads/),
加工产物落在 BERTC_DERIVED_ROOT(默认 data/derived/),两者都 gitignore。

part 数默认值 = v4-Large(HF 上的 Ismantic/BERTc-315M)实跑用量。
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 下载落地根。默认在仓库内,换机器不用改任何东西。
DATA_ROOT = Path(os.environ.get("BERTC_DATA_ROOT",
                                str(REPO_ROOT / "data" / "downloads")))
# 加工产物(documents.txt / jsonl / pkl)
DERIVED_ROOT = Path(os.environ.get("BERTC_DERIVED_ROOT",
                                   str(REPO_ROOT / "data" / "derived")))
# HF 镜像。置空走官方源。
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")


@dataclass
class Source:
    """一个可下载的数据源。

    kind:
      hf          HF 数据集,按 allow_patterns 选文件、按 n_parts 截断
      hf-snapshot HF 数据集,整仓下载(小仓库)
      github-file GitHub 单文件(raw.githubusercontent),zip 自动解压
      github-repo GitHub 仓库,git clone --depth 1
    """
    name: str
    kind: str
    repo_id: str
    subdir: str                    # 相对 DATA_ROOT
    part_glob: str                 # 相对 dir();github-file 时就是文件名
    n_parts: int | None = None     # None = 全量
    allow_patterns: list[str] = field(default_factory=list)
    note: str = ""

    def dir(self) -> Path:
        return DATA_ROOT / self.subdir

    def files(self, n: int | None = -1) -> list[Path]:
        """按 sorted glob 取前 n 个。n=-1 用注册的默认值,None 取全部。"""
        if n == -1:
            n = self.n_parts
        got = sorted(self.dir().glob(self.part_glob))
        return got if n is None else got[:n]

    def present(self) -> int:
        d = self.dir()
        return len(list(d.glob(self.part_glob))) if d.exists() else 0


# ---------------------------------------------------------------- 预训练语料
# v4-Large 实跑(17.65B token,中英 6:4):
#   wiki_cn 3 / PeopleDaily 1 / SkyPile 21 / CCI3-HQ 5 / FineWeb-Edu 500
#   CnnDM 1 / wiki_en 25

PRETRAIN_SOURCES = {
    "skypile": Source(
        name="skypile", kind="hf", repo_id="Skywork/SkyPile-150B",
        subdir="SkyPile", part_glob="*.parquet", n_parts=21,
        allow_patterns=["*.parquet"],
        note="中文网页。HF 上是 150B 全量,只取前 21 个 parquet。",
    ),
    "cci3": Source(
        name="cci3", kind="hf", repo_id="BAAI/CCI3-HQ",
        subdir="CCI3-HQ", part_glob="data/*.jsonl", n_parts=5,
        allow_patterns=["data/*.jsonl"],
        note="中文高质量语料,取前 5 个 jsonl。",
    ),
    "fineweb_edu_zh": Source(
        name="fineweb_edu_zh", kind="hf",
        repo_id="opencsg/Fineweb-Edu-Chinese-V2.2",
        subdir="Chinese-FineWeb-Edu-V2.2", part_glob="4_5/*.parquet", n_parts=500,
        allow_patterns=["4_5/*.parquet"],
        note="中文教育向。只用打分 4_5 的子集,取前 500 个(全量 9745)。",
    ),
    "finewiki_zh": Source(
        name="finewiki_zh", kind="hf", repo_id="HuggingFaceFW/finewiki",
        subdir="finewiki", part_glob="data/zhwiki/*.parquet", n_parts=None,
        allow_patterns=["data/zhwiki/*.parquet"],
        note="中文维基,5 parquet / 5.5GB。",
    ),
    "finewiki_en": Source(
        name="finewiki_en", kind="hf", repo_id="HuggingFaceFW/finewiki",
        subdir="finewiki", part_glob="data/enwiki/*.parquet", n_parts=None,
        allow_patterns=["data/enwiki/*.parquet"],
        note="英文维基,15 parquet / 37.7GB。",
    ),
    "people_daily": Source(
        name="people_daily", kind="hf-snapshot",
        repo_id="Papersnake/people_daily_news",
        subdir="people_daily_news", part_glob="*.jsonl.gz", n_parts=None,
        note="人民日报 1946-2025 全文。需 data/process.py 加工成 documents.txt。",
    ),
    "cnn_dailymail": Source(
        name="cnn_dailymail", kind="hf-snapshot", repo_id="abisee/cnn_dailymail",
        subdir="cnn_dailymail", part_glob="**/*.parquet", n_parts=None,
        note="英文新闻。同样需 data/process.py 加工。",
    ),
}

# ---------------------------------------------------------------- 下游任务

FINETUNE_SOURCES = {
    "pd1998": Source(
        name="pd1998", kind="github-file", repo_id="chenhui-bupt/PeopleDaily1998",
        subdir="PeopleDaily1998", part_glob="199801.zip", n_parts=None,
        note="PD-1998 PFR 标注语料。压缩包名叫 199801,里面其实是 "
             "199801.txt ~ 199806.txt 六个月。data/process_cws.py 解析成 jsonl。",
    ),
}

# CSC 的四个源。逐文件比对确认它们覆盖原 all_pairs.pkl 的全部内容;
# 早年 csc/data/raw 下的 sighan/ mcsc_full/ lemon_v2/ wang271k_raw/ 都是
# 这四个源的重复副本,ecspell/ cscd_ime/ 则是当年下载失败留下的 HTML 错误页。
CSC_SOURCES = {
    "ctc_dataset": Source(
        name="ctc_dataset", kind="github-repo", repo_id="zejunwang1/CTCDataset",
        subdir="csc/CTCDataset", part_glob="**/*.jsonl*", n_parts=None,
        note="中文文本纠错数据集汇总:CCTC / CTC2021 / MCSCSet / ECSpell / "
             "lemon / cscd-ns / sighan / yacsc / Wang271k。CSC 的主力,"
             "光 CTC2021/train_large_v2.jsonl.gz 就贡献 10 万对。",
    ),
    "mcscset": Source(
        name="mcscset", kind="github-repo", repo_id="yzhihao/MCSCSet",
        subdir="csc/MCSCSet", part_glob="**/annotated_data.txt", n_parts=None,
        note="医疗领域 CSC,专家标注 199,763 条。CTCDataset 里的 MCSCSet 是"
             "过滤后的子集,这份原始标注另有 5.9 万对独有内容。",
    ),
    "wang271k": Source(
        name="wang271k", kind="hf-snapshot", repo_id="shibing624/CSC",
        subdir="csc/wang271k", part_glob="*.json", n_parts=None,
        note="Wang271K + SIGHAN,MacBERT4CSC 的标准训练集,27.6 万对。",
    ),
    "sighan15_test": Source(
        name="sighan15_test", kind="github-file", repo_id="shibing624/pycorrector",
        subdir="csc/sighan15", part_glob="pycorrector/data/sighan2015_test.tsv",
        n_parts=None,
        note="SIGHAN-15 官方 707 条测试集(pycorrector vendored 的那份)。"
             "这是 CSC 的**权威基准**,报告的 F1 全部基于它 —— 注意 CTCDataset "
             "里的 sighan15_test.jsonl 是 1100 条的另一个版本,不能混用。",
    ),
    "chinese_text_correction": Source(
        name="chinese_text_correction", kind="hf-snapshot",
        repo_id="shibing624/chinese_text_correction",
        subdir="csc/shibing624", part_glob="*.tsv", n_parts=None,
        note="cscd_ns / medical_csc / lemon_* / ec_* 等 14 个 tsv。",
    ),
}

ALL_SOURCES = {**PRETRAIN_SOURCES, **FINETUNE_SOURCES, **CSC_SOURCES}

# ---------------------------------------------------------------- 加工产物

DERIVED = {
    "people_daily_docs": "PeopleDaily.documents.txt",
    "cnn_dailymail_docs": "CnnDailyMail.documents.txt",
}

PD98_DIR = DERIVED_ROOT / "pd98"                                   # cws/pos/ner jsonl
CSC_DIR = DERIVED_ROOT / "csc"
CSC_PAIRS = CSC_DIR / "all_pairs.pkl"                              # CSC 句对
SIGHAN_TEST = CSC_DIR / "sighan2015_test_official.tsv"             # 官方 707 条


def derived_path(key: str) -> Path:
    return DERIVED_ROOT / DERIVED[key]


def describe() -> str:
    lines = [f"DATA_ROOT    = {DATA_ROOT}",
             f"DERIVED_ROOT = {DERIVED_ROOT}",
             f"HF_ENDPOINT  = {HF_ENDPOINT or '(官方源)'}",
             "", "下载:"]
    for name, src in ALL_SOURCES.items():
        have = src.present()
        want = "全部" if src.n_parts is None else src.n_parts
        lines.append(f"  {'✓' if have else '✗'} {name:<24} {have:>5} 个(需 {want})"
                     f"  {src.repo_id}")
    lines.append("\n加工产物:")
    for key in DERIVED:
        p = derived_path(key)
        lines.append(f"  {'✓' if p.exists() else '✗'} {key:<24} {p}")
    for label, p in (("pd98 jsonl", PD98_DIR / "cws.pd98.jsonl"),
                     ("csc 句对", CSC_PAIRS),
                     ("sighan-15 测试集", SIGHAN_TEST)):
        lines.append(f"  {'✓' if p.exists() else '✗'} {label:<24} {p}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
