# BERTc 数据获取与加工
#
#   make status              看每个源下了没、加工了没
#   make download            下载全部(预训练约 79G + 下游约 1.5G)
#   make process             加工全部
#   make all                 下载 + 加工
#
# 分开跑:
#   make download-pretrain   只下预训练语料
#   make download-finetune   只下下游任务数据(PD-1998 需要先 clone Wapic,
#                            见 prepare/install_deps.sh wapic-data)
#   make download-skypile    只下某一个源(源名见 make status)
#   make process-docs        PeopleDaily / CnnDailyMail → documents.txt
#   make process-cws         PD-1998 → cws/pos/ner jsonl
#   make process-csc         CSC 各源 → 句对 pkl
#
# 落地位置由环境变量控制,默认在仓库内:
#   BERTC_DATA_ROOT=data/downloads    BERTC_DERIVED_ROOT=data/derived
#
# 下载走 hf-mirror(HF_ENDPOINT),脚本内部会清代理 —— hf-mirror 走不通代理,
# 而 GitHub 反过来需要代理,download.py 已经处理了这个矛盾。
# 走官方源:make download HF_ENDPOINT=

PY ?= /home/tfbao/.venv/bin/python
HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: all status download download-pretrain download-finetune download-csc \
        process process-docs process-cws process-csc clean-derived help

help:
	@awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' $(HERE)Makefile

all: download process

status:
	@$(PY) $(HERE)download.py --list

# ---------------------------------------------------------------- 下载

download: download-pretrain download-finetune

download-pretrain:
	$(PY) $(HERE)download.py --pretrain

download-finetune:
	$(PY) $(HERE)download.py --finetune

download-csc:
	$(PY) $(HERE)download.py --csc

# 单个源:make download-skypile / download-cci3 / download-finewiki_zh ...
download-%:
	$(PY) $(HERE)download.py $*

# ---------------------------------------------------------------- 加工

process: process-docs process-cws process-csc

# PeopleDaily(1946-2025 全文)/ CnnDailyMail → 一行一篇的 documents.txt,喂预训练
process-docs:
	$(PY) $(HERE)process.py --all

# PD-1998 的 PFR 标注 → 分词 / 词性 / 实体三套 jsonl,喂 MT 微调。
# 语料随 Wapic 仓库一起 clone,不单独下载。
process-cws:
	$(PY) $(HERE)process_cws.py

# CSC 四个源文件 → (错句, 正句) 对,249,975 对(已发布模型用的那份),
# 同时把 SIGHAN-15 官方 707 条拷到 derived/
process-csc:
	$(PY) $(HERE)process_csc.py

# ---------------------------------------------------------------- 清理

# 只删加工产物,下载的原件留着 —— 加工几分钟就能重来,重下要几小时
clean-derived:
	@echo "将删除 $${BERTC_DERIVED_ROOT:-$(HERE)derived}"
	@read -p "确认? [y/N] " a && [ "$$a" = y ]
	rm -rf $${BERTC_DERIVED_ROOT:-$(HERE)derived}
