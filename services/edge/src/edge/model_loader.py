import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from edge.config import PROFILES

"""
AutoModelForCausalLM: Loads the appropriate pretrained model
AutoTokenizer: Loads the tokenizer for that particular model 
BitsAndBytesConfig: Configures for quantization using the bitsandbytes module
"""

def load_model(profile_key: str):
    """
    :param profile_key: Key that is used to retrieve Model Profiles
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
        device_map={"": 0}, # If VRAM is sufficient, it loads entirely onto GPU else some layers in CPU. Here only GPU
    )
    tokenizer.pad_token = tokenizer.eos_token # make sure that all the tokens generated have the same
    return model, tokenizer, profile

def sanity_check(profile_key: str):
    model, tokenizer, profile = load_model(profile_key)
    prompt = "def fibonacci(n):\n    "
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=profile.max_new_tokens, do_sample=False)
    print(tokenizer.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    import sys
    sanity_check(sys.argv[1] if len(sys.argv) > 1 else "dev")
