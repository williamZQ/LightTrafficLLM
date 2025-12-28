import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.llama.modeling_llama import LlamaModel
from transformers import LlamaForCausalLM
from adapters import LoRALayer, Adapter


class GPT4TS(nn.Module):
    def __init__(self, device="cuda:0", gpt_layers=6, lora_rank=8, llm_path=None):
        super(GPT4TS, self).__init__()
        if llm_path is None:
            raise ValueError("llm_path must be provided")
        model_name = llm_path.rstrip('/').split('/')[-1]
        self.llm = None
        if 'llama' in model_name.lower():
            print("Loading Llama model from", llm_path)
            self.llm = LlamaModel.from_pretrained(
                llm_path, output_attentions=True, output_hidden_states=True
            )
            
            self.layers = self.llm.layers[:gpt_layers]
            model_type = "llama"
        elif 'qwen3' in model_name.lower():
            print("Loading Qwen3 model from", llm_path)
            self.llm = LlamaForCausalLM.from_pretrained(
                llm_path, output_attentions=True, output_hidden_states=True
            )
            
            self.layers = self.llm.model.layers[:gpt_layers]
            model_type = "qwen3"
        else:
            print("Loading GPT-2 model from", llm_path)
            self.llm = GPT2Model.from_pretrained(
                llm_path, output_attentions=True, output_hidden_states=True
            )
            
            self.layers = self.llm.h[:gpt_layers]
            model_type = "gpt2"
        
  
        for param in self.llm.parameters():
            param.requires_grad = False
        

        for i, layer in enumerate(self.layers):
            rank = lora_rank * (i + 1) // gpt_layers  # Higher rank for deeper layers
            
            if model_type == "llama" or model_type == "qwen3":
                # Apply LoRA to Llama/Qwen3 attention layers
                if hasattr(layer.self_attn, 'q_proj'):
                    layer.self_attn.q_proj = self._replace_with_lora(layer.self_attn.q_proj, rank)
                if hasattr(layer.self_attn, 'k_proj'):
                    layer.self_attn.k_proj = self._replace_with_lora(layer.self_attn.k_proj, rank)
                if hasattr(layer.self_attn, 'v_proj'):
                    layer.self_attn.v_proj = self._replace_with_lora(layer.self_attn.v_proj, rank)
                if hasattr(layer.self_attn, 'o_proj'):
                    layer.self_attn.o_proj = self._replace_with_lora(layer.self_attn.o_proj, rank)
                
                # Add adapter to Llama/Qwen3 layers
                layer.adapter = Adapter(layer.self_attn.head_dim)
            else:
                # Apply LoRA to GPT-2 attention layers
                if hasattr(layer.attn, 'c_attn'):
                    layer.attn.c_attn = self._replace_with_lora(layer.attn.c_attn, rank)
                if hasattr(layer.attn, 'c_proj'):
                    layer.attn.c_proj = self._replace_with_lora(layer.attn.c_proj, rank)
        
                layer.adapter = Adapter(layer.attn.embed_dim)
            
            if i >= gpt_layers - 2:
                for name, param in layer.named_parameters():
                    if "mlp" in name:
                        param.requires_grad = True
    
    def _replace_with_lora(self, original_layer, rank):
        lora_layer = LoRALayer(original_layer, rank=rank)
        return lora_layer

    def forward(self, x):
        return self.llm(inputs_embeds=x).hidden_states[-1]