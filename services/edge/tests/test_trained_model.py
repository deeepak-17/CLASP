from edge.model_loader import load_model
from edge.lora_init import attach_lora
from edge.toy.toy_training_loop import set_seed, train, toy_dataset, encode
from peft import PeftModel

def test_before_after_out():
    set_seed(0)
    model, tokenizer, profile = load_model("dev")
    target_modules = ("q_proj", "k_proj", "v_proj", "o_proj")
    model = attach_lora(model, r=16, target_modules=target_modules)

    examples = [encode(s, tokenizer) for s in toy_dataset]

    train(model, examples, epochs=10, lr=2e-4)
    model.save_pretrained(save_directory="./cache")
    tokenizer.save_pretrained(save_directory="./cache")

    # Switching to inference mode
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    prompt = "def fibonacci(n):\n    "
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs, max_new_tokens=profile.max_new_tokens, do_sample=False
    )
    before = out[0].tolist()

    base, tokenizer, _ = load_model("dev")
    model = PeftModel.from_pretrained(base, "./cache")
    model.eval()
    model.config.use_cache = True

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    after = out[0].tolist()
    assert before == after
