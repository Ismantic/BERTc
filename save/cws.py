"""交互式体验中文分词 + 词性标注 + 命名实体识别。

    python -m save.cws                              # 进 REPL
    cat corpus.txt | python -m save.cws -q          # 批量分词,一行一句
    python -m save.cws "中国科学院计算技术研究所在北京"  # 单句
    echo "..." | python -m save.cws                 # 管道

默认加载 save/releases/BERTc-315M-MT,用 --model 指向别的发布目录。

REPL 里可用:
    :s / :p / :n   只看分词 / 词性 / 实体
    :a             全部都看(默认)
    :q             退出
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "save" / "releases" / "BERTc-315M-MT"

# LTP base1 的 27 个词性代号,翻成中文 —— ns / ni / nh 这种没人记得住
POS_ZH = {
    "a": "形容词", "b": "区别词", "c": "连词", "d": "副词", "e": "叹词",
    "h": "前缀", "i": "成语", "j": "简称", "k": "后缀", "m": "数词",
    "n": "名词", "nd": "方位词", "nh": "人名", "ni": "机构名", "nl": "处所词",
    "ns": "地名", "nt": "时间词", "nz": "其他专名", "o": "拟声词", "p": "介词",
    "q": "量词", "r": "代词", "u": "助词", "v": "动词", "wp": "标点",
    "x": "非语素字", "z": "状态词",
}
NER_ZH = {"Nh": "人名", "Ns": "地名", "Ni": "机构名"}


class Style:
    def __init__(self, enabled: bool):
        self.on = enabled

    def _w(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.on else s

    def word(self, s):  return self._w("37;1", s)
    def pos(self, s):   return self._w("34", s)      # 蓝
    def ent(self, s):   return self._w("35;1", s)    # 紫
    def dim(self, s):   return self._w("2", s)
    def head(self, s):  return self._w("36;1", s)
    def warn(self, s):  return self._w("33", s)


def load(model_dir: Path):
    sys.path.insert(0, str(model_dir))
    from mt_model import BERTcForMT
    return BERTcForMT.from_pretrained(str(model_dir))


def render(r: dict, st: Style, show: str) -> str:
    text, words, pos, ents = r["text"], r["words"], r["pos"], r["ner"]
    lines = []

    if show in ("a", "s"):
        lines.append("  分词  " + st.dim(" / ").join(st.word(w) for w in words))

    if show in ("a", "p"):
        # 词/词性 挨着排,词性用中文,原代号跟在后面
        cells = [f"{st.word(w)}{st.pos('/' + POS_ZH.get(p, p))}"
                 for w, p in zip(words, pos)]
        lines.append("  词性  " + "  ".join(cells))

    if show in ("a", "n"):
        if ents:
            lines.append("  实体")
            for e in ents:
                span = text[e["start"]:e["end"]]
                kind = NER_ZH.get(e["type"], e["type"])
                lines.append(f"        {st.ent(f'[{kind}]')} {st.word(span)}"
                             f"  {st.dim(f'{e['start']}-{e['end']}')}")
        else:
            lines.append("  实体  " + st.dim("无"))

    # 整句视图:词间加 ·,实体整体染色,一眼看出切分和实体的关系
    if show == "a":
        ent_pos = {}
        for e in ents:
            for i in range(e["start"], e["end"]):
                ent_pos[i] = e["type"]
        pieces, cur = [], 0
        for w in words:
            seg = "".join(st.ent(c) if (cur + k) in ent_pos else c
                          for k, c in enumerate(w))
            pieces.append(seg)
            cur += len(w)
        lines.insert(0, "  " + st.dim("·").join(pieces))

    return "\n".join(lines)


def run_one(model, text: str, st: Style, show: str,
            quiet: bool = False) -> None:
    text = text.strip()
    if not text:
        return
    r = model.predict(text)
    if quiet:
        # 一行一句,空格分词 —— 接下游工具最省事的格式
        print(" ".join(r["words"]))
        return
    print(render(r, st, show))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="*", help="要分析的句子,不给就进 REPL")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--only", choices=("s", "p", "n"), default=None,
                    help="s 只分词 / p 只词性 / n 只实体")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="只输出空格分词结果,一行一句(适合管道)")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    st = Style(not args.no_color and sys.stdout.isatty())
    if not args.model.exists():
        sys.exit(f"找不到 {args.model} —— 先跑 python -m save.export BERTc-315M-MT")

    print(st.dim(f"加载 {args.model.name} ..."), file=sys.stderr)
    model = load(args.model)
    show = args.only or "a"

    if args.text:
        for t in args.text:
            run_one(model, t, st, show, args.quiet)
        return
    if not sys.stdin.isatty():
        for line in sys.stdin:
            run_one(model, line, st, show, args.quiet)
        return

    print(st.head("分词 + 词性 + 实体") +
          st.dim("  :s 分词  :p 词性  :n 实体  :a 全部  :q 退出"))
    while True:
        try:
            line = input(st.head("> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in (":q", ":quit", ":exit"):
            break
        if line in (":s", ":p", ":n", ":a"):
            show = line[1]
            label = {"s": "分词", "p": "词性", "n": "实体", "a": "全部"}[show]
            print(st.dim(f"  只看 {label}"))
            continue
        run_one(model, line, st, show)


if __name__ == "__main__":
    main()
