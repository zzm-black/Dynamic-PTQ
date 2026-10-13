import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForCausalLM, AutoConfig

def load_mixtral_model(model_path: str, attn_implementation: str = "eager"):
    config = AutoConfig.from_pretrained(model_path, attn_implementation=attn_implementation)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    model.seqlen = 2048
    num_layers = model.config.num_hidden_layers
    num_experts = model.config.num_local_experts
    return model, num_layers, num_experts

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if not layers:
        layers = [transformers.pytorch_utils.Conv1D, nn.Conv2d, nn.Linear]
    for layer in layers:
        if isinstance(module, layer):
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + "." + name1 if name != "" else name1))
    return res

