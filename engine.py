from pathlib import Path
import logging
import os
import threading
import time

LANGUAGES = ("Auto", "English", "Swahili", "Luganda", "Somali", "Kinyarwanda", "Dholuo", "Amharic")

def build_prompt(instruction, language):
    if language not in LANGUAGES:
        raise ValueError("Choose a supported language.")
    instruction = " ".join(instruction.split())
    if not instruction:
        raise ValueError("Enter a story instruction.")
    prompt = f"### Instruction: {instruction}"
    if language != "Auto":
        prompt += f" ### Target Language: {language}"
    return prompt + " ### Response:"

def validate_model_folder(folder):
    required = ("config.json", "model.safetensors", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json")
    missing = [name for name in required if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing files in {folder}:\n" + "\n".join(missing))
    for name in required:
        path = folder / name
        if path.stat().st_size == 0:
            raise ValueError(f"File is empty: {path}")
        with path.open("rb") as handle:
            header = handle.read(128)
        if header.startswith(b"version https://git-lfs.github.com/spec/"):
            raise RuntimeError(f"{name} is a Git LFS pointer, not the actual file. The deployment must download the LFS file contents before this model can load.")

class StoryEngine:
    def __init__(self, model_folder):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.lock = threading.Lock()
        folder = Path(model_folder).resolve()
        validate_model_folder(folder)
        logging.info("Loading custom tokenizer from %s", folder)
        self.tokenizer = AutoTokenizer.from_pretrained(str(folder), use_fast=False, local_files_only=True, trust_remote_code=False)
        if self.tokenizer.is_fast:
            raise RuntimeError("The original SentencePiece tokenizer is required.")
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer does not define an EOS token.")
        logging.info("Loading fine-tuned model from %s", folder)
        self.device = "cpu"
        torch.set_num_threads(min(4, os.cpu_count() or 2))
        self.model = AutoModelForCausalLM.from_pretrained(str(folder), torch_dtype=torch.float32, attn_implementation="eager", local_files_only=True, trust_remote_code=False)
        self.model.to(self.device).eval()
        self.context = int(self.model.config.max_position_embeddings)
        logging.info("Model ready: device=%s, context=%s", self.device, self.context)

    def generate(self, instruction, language, max_tokens=128, temperature=0.7, repetition_penalty=1.1):
        prompt = build_prompt(instruction, language)
        with self.lock:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
            count = inputs.input_ids.shape[1]
            available = self.context - count
            if available < 1:
                raise ValueError(f"Your instruction uses {count} tokens. The model's context limit is {self.context}. Please shorten the instruction.")
            budget = min(int(max_tokens), available)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            options = dict(max_new_tokens=budget, do_sample=temperature > 0, repetition_penalty=float(repetition_penalty), num_beams=1, eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.eos_token_id, use_cache=True)
            if temperature > 0:
                options.update(temperature=float(temperature), top_p=0.9, top_k=50)
            started = time.monotonic()
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, **options)
            generated = output[0, count:].detach().cpu().tolist()
            return dict(text=self.tokenizer.decode(generated, skip_special_tokens=True).strip(), prompt=prompt, input_tokens=count, output_tokens=len(generated), seconds=time.monotonic() - started, budget_reduced=budget < max_tokens)
