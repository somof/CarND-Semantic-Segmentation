PYTHON = ~/.pyenv/shims/python

UNAME = ${shell uname}
ifeq ($(UNAME),Darwin)
PYTHON = ../../../src/miniconda3/envs/IntroToTensorFlow/bin/python
endif

all:
	env LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64 $(PYTHON) main.py
