from transformers import AutoConfig
import torch
import torch.nn as nn
from typing import Optional
import os
from transformers.models.mistral.modeling_mistral import MistralModel
from transformers.models.mistral.modeling_mistral import MistralPreTrainedModel


class LLMForSequenceRegression(MistralModel):
    """
    Adapted from https://huggingface.co/kaist-ai/janus-rm-7b.
    """

    def __init__(self, config: AutoConfig):
        super().__init__(config)
        setattr(self, self.base_model_prefix, MistralPreTrainedModel(config))

        self.value_head = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
    ) -> torch.Tensor:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        outputs = getattr(self, self.base_model_prefix)(
            input_ids, attention_mask=attention_mask, position_ids=position_ids
        )
        last_hidden_states = outputs["last_hidden_state"]
        values = self.value_head(last_hidden_states).squeeze(-1)

        eos_indices = (
            attention_mask.size(1)
            - 1
            - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
        )
        reward = values.gather(dim=1, index=eos_indices).squeeze(1)

        if return_output:
            return reward, outputs
        else:
            return reward
