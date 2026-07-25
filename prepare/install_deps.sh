#!/usr/bin/env bash
# 重新编译安装 BERTc 的两个 C++ 依赖,并下载 Wapic 的分词模型。
#
#   bash prepare/install_deps.sh              # 装两个 + 下模型
#   bash prepare/install_deps.sh piece        # 只装 PieceTokenizer
#   bash prepare/install_deps.sh wapic        # 只装 Wapic(含模型)
#   bash prepare/install_deps.sh --verify     # 不装,只跑基线校验
#
# 装完会自动跑 tests/test_tokenizer.py 校验行为没变 —— 这一步不能跳:
#   - PieceTokenizer 的编码一旦变了,12536 词表和已发布模型的 embedding 就对不上,
#     但代码不会报错,只会悄悄训出/推出垃圾结果
#   - Wapic 的切词变了会改变 WWM 的词边界(影响预训练掩码粒度,不影响词表)
#
# 两个仓库都是 CMake 项目,需要 cmake、C++17 编译器。
# 用 uv pip,这个 venv 里没有 pip。
set -euo pipefail

PY=/home/tfbao/.venv/bin/python
PIECE_REPO=/home/tfbao/Shiyu/PieceTokenizer
WAPIC_REPO=/home/tfbao/Shiyu/Wapic
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TARGET="${1:-all}"

log() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

install_piece() {
    log "PieceTokenizer  ($PIECE_REPO)"
    [[ -d "$PIECE_REPO" ]] || { echo "找不到 $PIECE_REPO"; exit 1; }
    git -C "$PIECE_REPO" log -1 --format='  仓库 %h %ci  %s'
    # editable 安装:.so 会编译到仓库根目录,venv 通过 .pth 指过去。
    # --no-build-isolation 让它用 venv 里现成的 setuptools,省一次隔离环境构建。
    uv pip install -e "$PIECE_REPO" --python "$PY" --no-build-isolation --reinstall
}

install_wapic() {
    log "Wapic  ($WAPIC_REPO)"
    [[ -d "$WAPIC_REPO" ]] || { echo "找不到 $WAPIC_REPO"; exit 1; }
    git -C "$WAPIC_REPO" log -1 --format='  仓库 %h %ci  %s'
    # 非 editable:scikit-build-core 会把 _core.so 装进 site-packages/wapic/。
    # 注意这意味着**改了 Wapic 源码必须重跑这个脚本**,否则 venv 里还是旧的 ——
    # 以前是手工拷 .so,容易忘。
    uv pip install "$WAPIC_REPO" --python "$PY" --reinstall

    log "Wapic 分词模型"
    # 模型不再随仓库分发,从 HF 拉(Ismantic/wapic-cws → data/model/wapic-cws.wac)
    local model="$WAPIC_REPO/data/model/wapic-cws.wac"
    if [[ -f "$model" ]]; then
        echo "  已存在:$model  ($(du -h "$model" | cut -f1))"
    else
        # hf-mirror 走不通代理,必须清掉 —— Wapic 的 download.py 自己不清。
        # (GitHub 反过来需要代理,见 data/download.py 里的注释)
        (cd "$WAPIC_REPO" \
            && env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
                   -u all_proxy -u ALL_PROXY \
                   HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
               "$PY" scripts/download.py model)
    fi
}

case "$TARGET" in
    piece)    install_piece ;;
    wapic)    install_wapic ;;
    all)      install_piece; install_wapic ;;
    --verify) ;;
    *)        echo "用法: $0 [all|piece|wapic|--verify]"; exit 1 ;;
esac

log "校验:行为与 tests/fixtures/tokenizer_baseline.json 是否一致"
cd "$REPO_ROOT" && "$PY" tests/test_tokenizer.py
