from edge.model_loader import load_model
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

def attach_lora(model, r: int=8, lora_alpha: int=16, target_modules=('q_proj', 'v_proj'), dropout=0.0):
    """
    Prepares a base model for parameter-efficient fine-tuning via LoRA.

    :param model: The base HuggingFace CausalLM model instance.
    :param r: The rank of the LoRA update matrices (intrinsic dimensionality).
    :param lora_alpha: The scaling factor for the LoRA adapters.
    :param target_modules: Tuple/list of module names to apply LoRA layers to.
    :param dropout: The dropout probability for LoRA layers.
    :return: A PEFT-wrapped model ready for training.
    """
    # 1. Freezing: freezes the model parameters, upcasts to 16-bit or 32-bit in sensitive layers
    # use_reentrant=False: the reentrant autograd path for gradient checkpointing
    # is deprecated and a future torch makes the unset default an error. It also
    # interacts badly with frozen inputs, which is exactly our case.
    model = prepare_model_for_kbit_training(
        model, gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # 2. Configuration: Configuring LoRA hyper-parameters
    # W = W_base + (lora_alpha / r) * (A * B)
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        lora_dropout=dropout,
        task_type='CAUSAL_LM'
    )
    # 3. Wrapping: Wrap the base model with PEFT container
    model = get_peft_model(model, config)

    # 4. Printing: Print out trainable parameters
    layer = model.base_model.model.model.layers[0].self_attn.q_proj
    print("lora_A dtype:", layer.lora_A["default"].weight.dtype)
    print("lora_B dtype:", layer.lora_B["default"].weight.dtype)
    print("base weight dtype:", layer.base_layer.weight.dtype)  # should be uint8/int8-packed 4bit storage
    model.print_trainable_parameters() # trainable%: 0.1% to 1.0%
    return model

if __name__ == "__main__":
    """
    Pinned hyperparameters:
    r=16, lora_alpha=16, target_modules=(q_proj, k_proj, v_proj, o_proj), lora_dropout=0.0, task_type=CAUSAL_LM,
    base_model_id=deepseek-ai/deepseek-coder-1.3b-base
    """
    model, tokenizers, profile = load_model('dev')
    target_modules = ('q_proj', 'v_proj', 'k_proj', 'o_proj')
    attach_lora(model, target_modules=target_modules, r=16)
