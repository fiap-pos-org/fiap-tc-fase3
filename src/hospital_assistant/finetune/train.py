"""Fine-tuning QLoRA do Qwen2.5-1.5B-Instruct com o dataset do hospital.

Ajustado para GPU de 4 GB (RTX 3050 Ti): modelo em 4-bit (NF4), lote pequeno com acumulação
de gradiente (lote efetivo 16), gradient checkpointing e otimizador paginado em 8-bit. A loss é calculada só
sobre a resposta do assistente (o prompt não entra no gradiente).

Uso:
    uv run --group train python -m hospital_assistant.finetune.train --max-steps 20   # smoke test
    uv run --group train python -m hospital_assistant.finetune.train                  # treino completo
    uv run --group train python -m hospital_assistant.finetune.train --resume         # retoma
"""

import argparse
import json
import time

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from hospital_assistant.config import MODELS_DIR, PROCESSED_DIR

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = MODELS_DIR / "adapter"


def _prompt_completion(exemplo: dict) -> dict:
    """Converte o formato chat em prompt/completion: a loss fica só na completion."""
    return {"prompt": exemplo["messages"][:-1], "completion": exemplo["messages"][-1:]}


def _carrega_dataset():
    arquivos = {s: str(PROCESSED_DIR / f"{s}.jsonl") for s in ("train", "val")}
    ds = load_dataset("json", data_files=arquivos)
    colunas = ds["train"].column_names
    return ds.map(_prompt_completion, remove_columns=colunas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-steps", type=int, default=-1, help="limita passos (smoke test)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--output", default=str(ADAPTER_DIR))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        dtype=torch.bfloat16,
        device_map={"": 0},
    )

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,  # fração dos passos totais
        bf16=True,
        max_length=args.max_length,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        report_to="none",
        seed=42,
    )
    ds = _carrega_dataset()
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.model.print_trainable_parameters()

    torch.cuda.reset_peak_memory_stats()
    inicio = time.time()
    trainer.train(resume_from_checkpoint=args.resume or None)
    duracao = time.time() - inicio

    trainer.save_model(args.output)
    resumo = {
        "modelo_base": BASE_MODEL,
        "duracao_s": round(duracao),
        "vram_pico_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "passos": trainer.state.global_step,
        "args": vars(args),
        "log_history": trainer.state.log_history,
    }
    with open(f"{args.output}/treino.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)
    print(f"Treino concluído em {duracao / 60:.1f} min, pico de VRAM {resumo['vram_pico_gb']} GB")


if __name__ == "__main__":
    main()
