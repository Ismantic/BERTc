"""交互式体验中文拼写纠错。

    python -m save.csc                      # 进 REPL
    python -m save.csc "他平时喜欢锻练身体"   # 单句
    echo "..." | python -m save.csc         # 管道

默认加载 save/releases/BERTc-315M-CSC,用 --model 指向别的发布目录。

默认输出改动 + "存疑"(检测头报警但没纠成的位置)。
-v 额外显示置信度和检测分,-q 只输出纠正后的句子(适合管道)。

REPL 里可用:
    :t 0.5      改纠错阈值(默认 0.7,调低提召回、调高提精确率)
    :v          切换是否显示置信度和检测分
    :q          退出
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "save" / "releases" / "BERTc-315M-CSC"


class Style:
    """终端不是 tty 时(重定向、管道)自动降级成纯文本。"""

    def __init__(self, enabled: bool):
        self.on = enabled

    def _w(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.on else s

    def add(self, s):    return self._w("32;1", s)     # 绿:改后
    def rm(self, s):     return self._w("31;9", s)     # 红删除线:改前
    def dim(self, s):    return self._w("2", s)
    def head(self, s):   return self._w("36;1", s)     # 青
    def warn(self, s):   return self._w("33", s)


def load(model_dir: Path):
    sys.path.insert(0, str(model_dir))
    from csc_model import BERTcForCSC
    from tokenizer import PieceCharTokenizer
    model = BERTcForCSC.from_pretrained(str(model_dir))
    return model, PieceCharTokenizer(str(model_dir))


DET_ALERT = 0.5      # 检测分超过它就认为"模型觉得这里有问题"


def correct_verbose(model, tok, text: str, threshold: float, max_len: int = 128):
    """返回 (纠正后文本, 改动列表, 存疑列表)。

    比发布包的 correct() 多两样东西:

    - 每处改动的纠错置信度和检测分
    - **检测头报警但没纠成的位置**,连同 top-3 候选

    第二样才是关键。纠错只看纠错头的 argmax,检测头不参与,所以经常出现
    "模型知道这里有错、但选不出正确的字"—— 比如「我今天很稿兴」,稿 的
    检测分 0.98,可 top-1 仍是「稿」本身,「高」只排第 4。这种情况下调阈值
    没有任何用(top-1 是原字,阈值再低也换不掉),不显示出来的话
    使用者只会看到"没发现错误",完全摸不着头脑。
    """
    import torch

    device = next(model.parameters()).device
    input_ids, attn, lengths = tok.batch([text], max_len, device)
    with torch.no_grad():
        cor_logits, det_logits = model(input_ids, attn)
    probs = torch.softmax(cor_logits.float(), dim=-1)
    top_prob, top_id = probs.max(dim=-1)
    det = torch.sigmoid(det_logits.float())

    n = lengths[0]
    chars, changes, suspects = list(text[:n]), [], []
    for j in range(n):
        p, d = float(top_prob[0, j]), float(det[0, j])
        c = tok.id_to_char(int(top_id[0, j]))
        if p >= threshold and len(c) == 1 and c != chars[j]:
            changes.append((j, chars[j], c, p, d))
            chars[j] = c
        elif d >= DET_ALERT:
            top = probs[0, j].topk(3)
            cands = [(tok.id_to_char(int(i)), float(v))
                     for v, i in zip(*top)]
            suspects.append((j, text[j], d, cands))
    return "".join(chars) + text[n:], changes, suspects


def render(text: str, fixed: str, changes, suspects, st: Style,
           verbose: bool) -> str:
    lines = []
    if changes:
        # 把改动位置高亮出来,一眼能看到改了哪
        pos = {c[0] for c in changes}
        lines.append("  " + "".join(st.add(ch) if i in pos else ch
                                    for i, ch in enumerate(fixed)))
        for j, old, new, p, d in changes:
            detail = (f"   {st.dim(f'纠错置信度 {p:.3f}  检测分 {d:.3f}')}"
                      if verbose else "")
            lines.append(f"    第 {j + 1} 字  {st.rm(old)} → {st.add(new)}{detail}")
    else:
        lines.append(f"  {st.dim('没有改动')}")

    # 检测头报警但没纠成的位置。不显示的话使用者只会看到"没改动",
    # 不知道模型其实觉得这里有问题、只是选不出正确的字。
    if suspects:
        lines.append(f"  {st.warn('存疑')}  "
                     + st.dim("检测头报警但纠错头没选出别的字,调阈值无效"))
        for j, ch, d, cands in suspects:
            cand_s = "  ".join(f"{c}{st.dim(f'{v:.2f}')}" for c, v in cands)
            lines.append(f"    第 {j + 1} 字  {st.warn(ch)}  "
                         f"{st.dim(f'检测分 {d:.3f}')}  候选 {cand_s}")
    return "\n".join(lines)


def run_one(model, tok, text: str, st: Style, threshold: float,
            verbose: bool, quiet: bool = False):
    text = text.strip()
    if not text:
        return
    fixed, changes, suspects = correct_verbose(model, tok, text, threshold)
    if quiet:
        print(fixed)
        return
    print(f"  {st.dim(text)}")
    print(render(text, fixed, changes, suspects, st, verbose))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="*", help="要纠错的句子,不给就进 REPL")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="额外显示纠错置信度和检测分")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="只输出纠正后的句子,不要任何解释(适合管道)")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    st = Style(not args.no_color and sys.stdout.isatty())
    if not args.model.exists():
        sys.exit(f"找不到 {args.model} —— 先跑 python -m save.export BERTc-315M-CSC")

    print(st.dim(f"加载 {args.model.name} ..."), file=sys.stderr)
    model, tok = load(args.model)
    threshold, verbose = args.threshold, args.verbose

    if args.text:
        for t in args.text:
            run_one(model, tok, t, st, threshold, verbose, args.quiet)
        return
    if not sys.stdin.isatty():
        for line in sys.stdin:
            run_one(model, tok, line, st, threshold, verbose, args.quiet)
        return

    print(st.head("中文拼写纠错") +
          st.dim(f"  阈值 {threshold}  :t 改阈值  :v 置信度  :q 退出"))
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
        if line == ":v":
            verbose = not verbose
            print(st.dim(f"  置信度显示 {'开' if verbose else '关'}"))
            continue
        if line.startswith(":t"):
            try:
                threshold = float(line.split()[1])
                print(st.dim(f"  阈值 → {threshold}"))
            except (IndexError, ValueError):
                print(st.warn("  用法::t 0.5"))
            continue
        run_one(model, tok, line, st, threshold, verbose)


if __name__ == "__main__":
    main()
