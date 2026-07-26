# BERTc 发布层:checkpoint → 发布目录 → Hugging Face
#
#   make status              哪些 checkpoint 就位、哪些已导出
#   make export              导出全部六个发布目录
#   make code                只刷新已有发布目录里的代码和模型卡,不碰权重
#   make verify              发布目录能否独立跑(切进目录,只用目录内的模块)
#   make dry-run             看要上传什么,不真传
#   make upload              上传(权重 + 代码)
#   make upload-code         只上传代码和模型卡,权重不动
#
# 单个仓库:make export NAME=BERTc-315M-MT
# 换 namespace:make upload NS=someone
#
# 上传要 write token。**不要写进文件或命令行**,用交互登录:
#   huggingface-cli login
#
# 改了 src/model.py、save/assets/ 或模型卡之后的标准流程:
#   make code && make verify && make dry-run && make upload-code

HERE := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
ROOT := $(patsubst %/,%,$(dir $(HERE)))

-include $(ROOT)/local.mk
PY ?= $(if $(VIRTUAL_ENV),$(VIRTUAL_ENV)/bin/python,python3)

NS   ?= Ismantic
NAME ?=

.PHONY: help status export code verify dry-run upload upload-code

help:
	@awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' $(HERE)/Makefile

status:
	@cd $(ROOT) && $(PY) -m save.export --list

# ---------------------------------------------------------------- 导出

export:
	cd $(ROOT) && $(PY) -m save.export $(NAME)

# 权重没变、只改了推理代码或模型卡时用这个 —— 不重写 model.safetensors,
# 也就不需要原始 checkpoint 在场
code:
	cd $(ROOT) && $(PY) -m save.export --code-only $(NAME)

# 切到发布目录内 import 并推理,只能看到目录内的模块 —— 跟外部用户的处境一样
verify:
	cd $(ROOT) && $(PY) test/test_save.py

# ---------------------------------------------------------------- 上传

dry-run:
	cd $(ROOT) && $(PY) -m save.upload --namespace $(NS) --dry-run $(NAME)

upload: verify
	cd $(ROOT) && $(PY) -m save.upload --namespace $(NS) $(NAME)

upload-code: verify
	cd $(ROOT) && $(PY) -m save.upload --namespace $(NS) --code-only $(NAME)
