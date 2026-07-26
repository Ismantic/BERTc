#!/usr/bin/env bash
# 装 BERTc 的两个 C++ 依赖:PieceTokenizer(字级分词器 + 词表)和 Wapic(CRF 分词器)。
# 仓库不在本地就从 GitHub clone —— 整个 BERTc 只依赖 Hugging Face 和 GitHub。
#
#   bash prepare/install_deps.sh              # 两个都装 + 下 Wapic 模型
#   bash prepare/install_deps.sh piece        # 只装 PieceTokenizer
#   bash prepare/install_deps.sh wapic        # 只装 Wapic(含模型)
#   bash prepare/install_deps.sh wapic-data   # 只 clone Wapic,不编译不下模型
#   bash prepare/install_deps.sh --verify     # 不装,只跑行为校验
#
# wapic-data 是给只做微调的人用的:PD-1998 标注语料在 Wapic 仓库的
# data/PeopleDaily1998.zip 里,拿它只需要 clone,不需要编译分词器。
#
# 仓库默认 clone 到 BERTC_DEPS_DIR(默认 <仓库>/deps),已有就 git pull。
# 想用本机既有的 checkout:BERTC_DEPS_DIR=/home/tfbao/Shiyu bash prepare/install_deps.sh
#
# 装完自动跑 test/test_tokenizer.py 校验行为没变 —— 这一步不能跳:
#   - PieceTokenizer 的编码一旦变了,12536 词表和已发布模型的 embedding 就对不上,
#     但代码不会报错,只会悄悄训出/推出垃圾结果
#   - Wapic 的切词变了会改变 WWM 的词边界(影响预训练掩码粒度,不影响词表)
#
# 需要 cmake、C++17 编译器、git。用 uv pip,这个 venv 里没有 pip。
set -euo pipefail

PY=${BERTC_PYTHON:-/home/tfbao/.venv/bin/python}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_DIR=${BERTC_DEPS_DIR:-$REPO_ROOT/deps}

PIECE_REPO=$DEPS_DIR/PieceTokenizer
WAPIC_REPO=$DEPS_DIR/Wapic
PIECE_URL=https://github.com/Ismantic/PieceTokenizer.git
WAPIC_URL=https://github.com/Ismantic/Wapic.git

TARGET="${1:-all}"

log() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# GitHub 要走代理,hf-mirror 反过来要清代理(见 data/download.py)
git_clone_or_pull() {
    local url=$1 dst=$2
    if [[ -d "$dst/.git" ]]; then
        echo "  已有 $dst,git pull"
        git -C "$dst" pull --ff-only
    else
        mkdir -p "$(dirname "$dst")"
        git clone --depth 1 "$url" "$dst"
    fi
    git -C "$dst" log -1 --format='  %h %ci  %s'
}

install_piece() {
    log "PieceTokenizer"
    git_clone_or_pull "$PIECE_URL" "$PIECE_REPO"
    # editable 安装:.so 编译到仓库根目录,venv 通过 .pth 指过去。
    # 这样 prepare/tokenizer.py 能顺着 piece_tokenizer.__file__ 找到同仓库的
    # save/BERTc-Tokenizer.pt —— 词表只有一个来源,不会两边漂移。
    uv pip install -e "$PIECE_REPO" --python "$PY" --no-build-isolation --reinstall
    local model="$PIECE_REPO/save/BERTc-Tokenizer.pt"
    [[ -f "$model" ]] && echo "  词表 $model ($(du -h "$model" | cut -f1))" \
                      || echo "  !! 仓库里没有 save/BERTc-Tokenizer.pt"
}

install_wapic() {
    log "Wapic"
    git_clone_or_pull "$WAPIC_URL" "$WAPIC_REPO"
    # 非 editable:scikit-build-core 把 _core.so 装进 site-packages/wapic/。
    # 也就是说**改了 Wapic 源码必须重跑本脚本**,否则 venv 里还是旧的。
    uv pip install "$WAPIC_REPO" --python "$PY" --reinstall

    log "Wapic 分词模型"
    local model="$WAPIC_REPO/data/model/wapic-cws.wac"
    if [[ -f "$model" ]]; then
        echo "  已存在:$model ($(du -h "$model" | cut -f1))"
    else
        # hf-mirror 走不通代理,必须清掉 —— Wapic 的 download.py 自己不清
        (cd "$WAPIC_REPO" \
            && env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
                   -u all_proxy -u ALL_PROXY \
                   HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
               "$PY" scripts/download.py model)
    fi
    echo "  BERTC_WAPIC_MODEL=$model"
}

# 只 clone,不 pip install、不下模型。给只做微调的人拿 PD-1998 语料用。
install_wapic_data() {
    log "Wapic 仓库(只 clone)"
    git_clone_or_pull "$WAPIC_URL" "$WAPIC_REPO"
    local zip="$WAPIC_REPO/data/PeopleDaily1998.zip"
    [[ -f "$zip" ]] && echo "  PD-1998 $zip ($(du -h "$zip" | cut -f1))" \
                    || { echo "  !! 仓库里没有 data/PeopleDaily1998.zip"; exit 1; }
}

case "$TARGET" in
    piece)      install_piece ;;
    wapic)      install_wapic ;;
    wapic-data) install_wapic_data ;;
    all)        install_piece; install_wapic ;;
    --verify)   ;;
    *)          echo "用法: $0 [all|piece|wapic|wapic-data|--verify]"; exit 1 ;;
esac

# wapic-data 没装任何东西,没什么可校验的
if [[ "$TARGET" != "wapic-data" ]]; then
    log "校验:行为与 test/fixtures/tokenizer_baseline.json 是否一致"
    cd "$REPO_ROOT" && "$PY" test/test_tokenizer.py
fi
