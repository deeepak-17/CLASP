"""Load the frozen NF4 base model for a profile (P1 Edge).

AutoModelForCausalLM: loads the appropriate pretrained model.
AutoTokenizer: loads the tokenizer for that particular model.
BitsAndBytesConfig: configures 4-bit quantization through bitsandbytes.

Non-quantized modules (embeddings, ``lm_head``, norms) still load in fp32: no
``dtype`` is passed to ``from_pretrained``. Passing ``torch.bfloat16`` would
save ~0.27 GB on the 1.3B profile, but it changes the numerics every recorded
perplexity was measured under (round 1, D3 round 2), so it waits until a
deliberate re-baseline rather than drifting in silently.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from edge.config import PROFILES

#: Whole model on GPU 0. Parameterized so a CPU-offload fallback can pass e.g.
#: "auto" without editing this module.
DEFAULT_DEVICE_MAP = {"": 0}


def load_model(profile_key: str, device_map=None):
    """
    :param profile_key: Key that is used to retrieve Model Profiles
    :param device_map: transformers device_map; defaults to the whole model on GPU 0
    :return: model, tokenizer, profile
    """
    profile = PROFILES[profile_key] # Loads the requested model profile

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, # The weights are converted to 4 bits numbers
        bnb_4bit_quant_type="nf4", # The quantization used if Normal Float 4 (better accuracy than INT4)
        bnb_4bit_compute_dtype=torch.bfloat16, # Converts to Brain Floating 16 only during computation
        bnb_4bit_use_double_quant=True, # Applies quantization to the scaling factor too.
    )

    tokenizer = AutoTokenizer.from_pretrained(profile.model_id) # Loads the tokenizer of the resp model
    model = AutoModelForCausalLM.from_pretrained( # Loads the pretrained model based on the configuration
        profile.model_id,
        quantization_config=bnb_config, # Applies the quantization; instead of FP16, loads NF4
        device_map=DEFAULT_DEVICE_MAP if device_map is None else device_map,
    )
    tokenizer.pad_token = tokenizer.eos_token # make sure that all the tokens generated have the same
    return model, tokenizer, profile

def sanity_check(profile_key: str):
    model, tokenizer, profile = load_model(profile_key)
    prompt = "def fibonacci(n):\n    "
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=profile.max_new_tokens, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id)
    print(tokenizer.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    import sys
    sanity_check(sys.argv[1] if len(sys.argv) > 1 else "dev")
