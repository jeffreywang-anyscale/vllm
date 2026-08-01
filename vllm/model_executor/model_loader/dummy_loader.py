# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch.nn as nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.reload.layerwise import (
    _get_original_loader,
    get_layerwise_info,
)
from vllm.model_executor.model_loader.reload.meta import materialize_layer
from vllm.model_executor.model_loader.reload.types import LayerReloadingInfo
from vllm.model_executor.model_loader.reload.utils import get_layer_tensors
from vllm.model_executor.model_loader.weight_utils import (
    initialize_dummy_weights,
    initialize_single_dummy_weight,
)


class DummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        for layer in model.modules():
            info = get_layerwise_info(layer)
            if info.can_load():
                self._process_online_quant_layer(layer, info)
            else:
                # NOTE(woosuk): For accurate performance evaluation, we assign
                # random values to the weights.
                initialize_dummy_weights(layer, model_config)

        # Keep dummy loading consistent with checkpoint loading for models
        # that require a model-level post-load weight transformation. DeepSeek
        # V4 MegaMoE, for example, packs its expert weights from its real
        # ``load_weights`` implementation. Without this call, the first
        # profile forward performs that packing while profiling is already
        # holding its activation allocations, producing an artificial OOM.
        #
        # Some model wrappers expose the finalizer on the outer module while
        # DeepSeek V4 exposes it on ``model.model``. Invoke at most one hook:
        # a wrapper-level hook is responsible for delegating to its inner model.
        for candidate in (model, getattr(model, "model", None)):
            finalizer = getattr(candidate, "finalize_mega_moe_weights", None)
            if callable(finalizer):
                finalizer()
                break

    def _process_online_quant_layer(
        self,
        layer: nn.Module,
        info: LayerReloadingInfo,
    ) -> None:
        """Materialize, apply dummy weights, and run quantization processing."""
        materialize_layer(layer, info)

        for tensor in get_layer_tensors(layer).values():
            initialize_single_dummy_weight(tensor)

        for param in get_layer_tensors(layer).values():
            param.weight_loader = _get_original_loader(param)

        quant_method = getattr(layer, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            quant_method.process_weights_after_loading(layer)

        info.reset()
