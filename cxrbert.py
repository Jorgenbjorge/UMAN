#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor as T
from transformers import BertConfig, BertTokenizer, BertPreTrainedModel
from transformers.models.bert.modeling_bert import *
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput


_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"


# https://github.com/microsoft/hi-ml/blob/c606808b20c88d6e2cc388bd650abf34ccae17cf/hi-ml-multimodal/src/health_multimodal/text/model/configuration_cxrbert.py#L25
class CXRBertTokenizer(BertTokenizer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

# --- 删除了硬编码的 TOKENIZER 加载 (以避免路径冲突) ---
# 原代码: tokenizer = CXRBertTokenizer.from_pretrained("BiomedVLP-CXR-BERT-specialized")


class CXRBertConfig(BertConfig):
    """
    Config class for CXR-BERT model.

    :param projection_size: Dimensionality of the joint latent space.
    """

    model_type = "cxr-bert"

    def __init__(self, projection_size: int = 128, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.projection_size = projection_size

@dataclass
class CXRBertOutput(ModelOutput):
    """
    Base class for outputs of CXRBertModel.

    :param loss: Classification loss
    :param logits: Classification scores
    :param cls_projected_embedding: CLS token embedding, after projection layer, possibly normalized.
    """

    loss: Optional[T] = None
    logits: T = None
    hidden_states: Optional[Tuple[T, ...]] = None
    attentions: Optional[Tuple[T, ...]] = None
    cls_projected_embedding: Optional[T] = None


class CXRBertModel(BertPreTrainedModel):
    """
    CXRBERT model with:
    - A projection head for joint vision-language embedding (CLS token)
    - A Masked Language Modeling head for MLM training
    """

    config_class = CXRBertConfig
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True

    def __init__(self, config: CXRBertConfig):
        super().__init__(config)

        # Backbone: standard BERT (without pooler)
        self.bert = BertModel(config, add_pooling_layer=False)

        # Projection head for CLS token (for contrastive / ITM)
        self.proj_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Linear(config.hidden_size, config.projection_size),
        )

        # ✅ MLM head (same as in BertForMaskedLM)
        self.mlm_head = BertLMPredictionHead(config)

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_cls_projected_embedding: bool = False,
        output_mlm_logits: bool = False,  # ✅ 新增
    ) -> Union[Tuple[torch.Tensor, ...], CXRBertOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Standard BERT forward
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # (batch, seq_len, hidden)
        sequence_output = outputs[0]
        cls_output = sequence_output[:, 0, :]

        # Projected CLS token (for CLIP/ITM)
        cls_projected_embedding = self.proj_head(cls_output)

        # ✅ MLM logits (token-level prediction)
        prediction_scores = None
        if output_mlm_logits:
            prediction_scores = self.mlm_head(sequence_output)

        if not return_dict:
            output = (cls_projected_embedding, prediction_scores) + outputs[1:]
            return output

        return CXRBertOutput(
            cls_projected_embedding=cls_projected_embedding if output_cls_projected_embedding else None,
            logits=prediction_scores if output_mlm_logits else None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_projected_text_embeddings(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, normalize_embeddings: bool = True
    ) -> torch.Tensor:
        """
        Returns l2-normalised projected CLS token embeddings for the given input token ids and attention mask.
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_cls_projected_embedding=True,
            return_dict=True
        )
        cls_projected_embedding = outputs.cls_projected_embedding
        if normalize_embeddings:
            return F.normalize(cls_projected_embedding, dim=1)
        return cls_projected_embedding


if __name__ == "__main__":
    # Example usage (needs model/tokenizer files to run)
    try:
        # Example loading from a placeholder path
        model = CXRBertModel.from_pretrained("./test_path") 
        input_ids = torch.randint(0, 512, (4, 20))
        attention_mask = torch.ones((4, 20))
        outputs = model(input_ids, attention_mask=attention_mask, output_cls_projected_embedding=True, return_dict=True)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Example usage failed (expected if path is not real): {e}")
