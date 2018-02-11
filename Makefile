PYTHON = env LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64 ~/.pyenv/versions/3.5.4/bin/python
PYTHON = env LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64 ~/.pyenv/shims/python
PYTHON35 = ~/.pyenv/versions/3.5.4/bin/python
PYTHON36 = ~/.pyenv/versions/3.6.4/bin/python
OPTIMIZE = ../tensorflow/tensorflow/python/tools/optimize_for_inference.py
TRANSFORM = ../tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph

define OPT_OPTIONS_FAIL
add_default_attributes \
remove_nodes(op=Identity, op=CheckNumerics) \
fold_constants(ignore_errors=true) \
fold_batch_norms \
fold_old_batch_norms \
fuse_resize_and_conv \
quantize_weights \
quantize_nodes \
strip_unused_nodes \
sort_by_execution_order
endef

define OPT_OPTIONS
add_default_attributes \
remove_nodes(op=Identity, op=CheckNumerics) \
fold_constants(ignore_errors=true) \
fold_batch_norms \
fold_old_batch_norms \
fuse_resize_and_conv \
strip_unused_nodes \
sort_by_execution_order
endef

all: optimize infer

infer:
	$(PYTHON36) reuse_graph_video.py | tee ss-infer.log
	@#$(PYTHON36) reuse_graph.py | tee ss-infer.log
	@#$(PYTHON36) reuse_graph_video.py | tee ss-infer.log
	@#$(PYTHON) reuse_graph_video.py | tee ss-infer-gpu.log

train:
	$(PYTHON35) main.py | tee ss-gpu-train.log
	@#$(PYTHON) main.py | tee ss-gpu-train.log

optimize:
	$(PYTHON) $(OPTIMIZE) \
	--input=runs/frozen_graph.pb \
	--output=runs/optimized_graph.pb \
	--frozen_graph=True \
	--input_names=image_input \
	--output_names=adam_logit;
	$(TRANSFORM) \
	--in_graph=runs/optimized_graph.pb \
	--out_graph=runs/eightbit_graph.pb \
	--inputs=image_input \
	--outputs=adam_logit \
	--transforms='${OPT_OPTIONS}';
	\mv -f runs/optimized_graph.pb runs_reuse/;
	\mv -f runs/eightbit_graph.pb runs_reuse/;
