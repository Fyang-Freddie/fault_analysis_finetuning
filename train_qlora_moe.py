#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯文本 MoE 模型 QLoRA 训练脚本。

默认模型：
    Qwen/Qwen3-30B-A3B

数据格式与 train_qlora.py 相同：
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "背景知识与检测事实..."},
    {"role": "assistant", "content": "<think>...</think>\n最终答案"}
  ]
}

与稠密模型脚本的主要区别：
1. MoE 的普通注意力层、共享专家层仍是 nn.Linear，可以按模块名称注入 LoRA。
2. Qwen3-MoE 的路由专家权重是三维 nn.Parameter，不会被 all-linear 自动覆盖。
3. 默认不训练路由专家，只训练注意力和共享专家，以减少显存、训练时间和过拟合风险。
4. 如确实需要训练路由专家，可传入 --train_experts；这要求支持 target_parameters
   的较新版本 PEFT，并会明显增加 adapter 大小和运行开销。

基础用法：
python train_qlora_moe.py \
  --model_name_or_path Qwen/Qwen3-30B-A3B \
  --train_file data/train.jsonl \
  --validation_file data/val.jsonl \
  --output_dir outputs/qwen3_30b_a3b_qlora

同时训练路由专家：
python train_qlora_moe.py \
  --model_name_or_path Qwen/Qwen3-30B-A3B \
  --train_experts \
  --output_dir outputs/qwen3_30b_a3b_expert_qlora

注意：
- 本脚本只处理纯文本 MoE 因果语言模型。
- Qwen/Qwen3.5-35B-A3B 带视觉编码器，需要 AutoProcessor 和多模态模型类，
  不能直接使用本脚本。
- Qwen3-MoE 至少需要 transformers>=4.51.0。
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# 复用已经验证过的数据校验、assistant-only mask 和动态 padding 逻辑，
# 避免在两个训练脚本中维护两份完全相同的实现。
from train_qlora import (
    AssistantOnlyDataCollator,
    build_preprocess_function,
    validate_jsonl,
)

LOGGER = logging.getLogger(__name__)


# Qwen3-MoE 中普通注意力和共享专家使用的线性模块名称。
# 这里不使用 all-linear，避免换成带视觉塔的模型时误训练视觉层。
DEFAULT_TARGET_MODULES = (
    "q_proj,k_proj,v_proj,o_proj,"
    "gate_proj,up_proj,down_proj"
)

# Qwen3-MoE 路由专家的权重不是 nn.Linear，而是三维 nn.Parameter。
# PEFT 需要通过 target_parameters 显式定位这些参数。
EXPERT_TARGET_PARAMETERS = (
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Text-only MoE causal-LM QLoRA training"
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-30B-A3B",
        help="Hugging Face 纯文本 MoE 因果语言模型名称或本地模型目录。",
    )
    parser.add_argument(
        "--model_revision",
        type=str,
        default=None,
        help="可选：模型分支、标签或 commit ID；正式训练建议固定 commit ID。",
    )
    parser.add_argument("--train_file", type=str, default="data/train.jsonl")
    parser.add_argument("--validation_file", type=str, default="data/val.jsonl")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/qwen3_30b_a3b_qlora",
    )

    # 30B 总权重即使使用 4-bit 也需要较大显存；默认长度比 8B 脚本更保守。
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        type=str,
        default=DEFAULT_TARGET_MODULES,
        help=(
            "逗号分隔的普通线性模块名称。默认覆盖注意力和共享专家，"
            "不使用可能误命中其他子模型的 all-linear。"
        ),
    )
    parser.add_argument(
        "--train_experts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "是否同时对三维路由专家权重注入 LoRA。默认关闭；"
            "开启后 adapter、显存和计算开销都会增加。"
        ),
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="例如 outputs/.../checkpoint-60。",
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "是否执行模型仓库中的自定义 Python 代码。默认关闭；"
            "仅对可信模型仓库显式开启。"
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use_double_quant",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--train_on_inputs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="默认关闭，只对最后一条 assistant 输出计算 loss。",
    )
    parser.add_argument(
        "--save_merged_model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "训练后尝试合并 LoRA。30B MoE 合并需要大量 CPU/GPU 内存，"
            "默认只保存 adapter。"
        ),
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """在下载大模型之前检查容易写错的数值参数。"""
    if args.max_length <= 0:
        raise ValueError("--max_length 必须大于 0。")
    if args.lora_r <= 0:
        raise ValueError("--lora_r 必须大于 0。")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("--lora_dropout 必须位于 [0, 1) 范围内。")


def resolve_device_map() -> Dict[str, int]:
    """
    单进程或 accelerate 数据并行时，将一整份 4-bit 模型放到当前进程的 GPU。

    这属于数据并行：每张 GPU 都会加载一份完整模型，不是模型切分。
    如果单卡无法容纳模型，应改用 FSDP/DeepSpeed，而不是 device_map="auto" 训练。
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return {"": local_rank}


def parse_target_modules(value: str) -> List[str]:
    """把逗号分隔参数转换为 PEFT 所需的模块名称列表。"""
    modules = [module.strip() for module in value.split(",") if module.strip()]
    if not modules:
        raise ValueError("--target_modules 不能为空。")
    if "all-linear" in modules:
        raise ValueError(
            "MoE 脚本不接受 all-linear；请显式列出语言模块，避免误训练其他子模型。"
        )
    return modules


def build_lora_config(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> tuple[LoraConfig, List[str], int | None]:
    """
    构造 MoE LoRA 配置。

    普通 nn.Linear 通过 target_modules 定位；只有显式开启 --train_experts 时，
    才通过 target_parameters 对三维专家权重注入 LoRA。
    """
    target_modules = parse_target_modules(args.target_modules)
    lora_kwargs: Dict[str, Any] = {}
    expert_targets: List[str] = []
    expert_rank: int | None = None

    if args.train_experts:
        # target_parameters 从 PEFT 0.17 起提供；提前检查可给出比构造器更清晰的错误。
        if "target_parameters" not in inspect.signature(LoraConfig).parameters:
            raise RuntimeError(
                "--train_experts 需要支持 target_parameters 的新版 PEFT，"
                "请升级到 peft>=0.17.0。"
            )

        parameter_names = [name for name, _ in model.named_parameters()]
        expert_targets = [
            suffix
            for suffix in EXPERT_TARGET_PARAMETERS
            if any(name.endswith(suffix) for name in parameter_names)
        ]
        if len(expert_targets) != len(EXPERT_TARGET_PARAMETERS):
            raise ValueError(
                "当前模型没有找到预期的 Qwen3-MoE 专家参数："
                f"{', '.join(EXPERT_TARGET_PARAMETERS)}。"
                "如使用其他 MoE 架构，请按其参数名称调整 EXPERT_TARGET_PARAMETERS。"
            )

        # 专家数量越多，直接给每个专家使用完整 lora_r 的 adapter 就越大。
        # 参考 PEFT 的 MoE 建议，将专家 rank 按专家数缩小，最低保留 1。
        config = getattr(model, "config", None)
        text_config = getattr(config, "text_config", config)
        num_experts = getattr(text_config, "num_experts", None)
        if not isinstance(num_experts, int) or num_experts <= 0:
            raise ValueError("无法从模型配置中读取有效的 num_experts。")
        expert_rank = max(1, args.lora_r // num_experts)

        lora_kwargs["target_parameters"] = expert_targets
        lora_kwargs["rank_pattern"] = {
            "experts.gate_up_proj": expert_rank,
            "experts.down_proj": expert_rank,
        }

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        **lora_kwargs,
    )
    return peft_config, expert_targets, expert_rank


def main() -> None:
    args = parse_args()
    validate_args(args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("该脚本需要 NVIDIA GPU 和 CUDA 环境。")
    if not torch.cuda.is_bf16_supported():
        LOGGER.warning("当前 GPU 可能不原生支持 BF16，将改用 FP16 计算。")

    validate_jsonl(args.train_file)
    validate_jsonl(args.validation_file)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading tokenizer: %s", args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "当前 tokenizer 没有 chat_template，请使用 Instruct/Chat MoE 模型。"
        )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer 同时缺少 pad_token_id 与 eos_token_id。")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=args.use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    LOGGER.info("Loading 4-bit MoE model. This can take several minutes.")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        quantization_config=quantization_config,
        dtype=compute_dtype,
        device_map=resolve_device_map(),
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    )

    model_type = getattr(model.config, "model_type", "")
    if "moe" not in model_type.lower():
        raise ValueError(
            f"模型类型 {model_type!r} 看起来不是 MoE；"
            "稠密模型请使用 train_qlora.py。"
        )

    model.config.use_cache = False
    if hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1

    # 先冻结并准备 4-bit 基础模型，再注入可训练 LoRA 参数。
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )
    peft_config, expert_targets, expert_rank = build_lora_config(model, args)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if args.train_experts:
        LOGGER.info(
            "Expert LoRA enabled: targets=%s, expert_rank=%d",
            expert_targets,
            expert_rank,
        )
    else:
        LOGGER.info(
            "Expert tensors are frozen; training attention and shared-expert modules only."
        )

    raw_datasets = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            "validation": args.validation_file,
        },
    )
    preprocess = build_preprocess_function(
        tokenizer=tokenizer,
        max_length=args.max_length,
        train_on_inputs=args.train_on_inputs,
    )
    tokenized_datasets = raw_datasets.map(
        preprocess,
        remove_columns=raw_datasets["train"].column_names,
        desc="Applying chat template and masking prompt tokens",
        num_proc=1,
    )

    def has_trainable_token(example: Dict[str, List[int]]) -> bool:
        return any(label != -100 for label in example["labels"])

    tokenized_datasets = tokenized_datasets.filter(
        has_trainable_token,
        desc="Removing samples without assistant labels",
    )
    LOGGER.info(
        "Tokenized samples: train=%d, validation=%d",
        len(tokenized_datasets["train"]),
        len(tokenized_datasets["validation"]),
    )
    if len(tokenized_datasets["train"]) == 0:
        raise RuntimeError("训练集经过处理后为空。")
    if len(tokenized_datasets["validation"]) == 0:
        raise RuntimeError("验证集经过处理后为空。")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=(compute_dtype == torch.bfloat16),
        fp16=(compute_dtype == torch.float16),
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        # Transformers 5.x 用 train_sampling_strategy 取代了 group_by_length。
        train_sampling_strategy="group_by_length",
        ddp_find_unused_parameters=False,
        seed=args.seed,
        data_seed=args.seed,
    )

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=AssistantOnlyDataCollator(tokenizer=tokenizer),
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    LOGGER.info("Starting MoE QLoRA training")
    train_result = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    LOGGER.info("Saving best LoRA adapter to %s", output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    train_metrics = train_result.metrics
    train_metrics["train_samples"] = len(tokenized_datasets["train"])
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(tokenized_datasets["validation"])
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    run_config = vars(args).copy()
    run_config["compute_dtype"] = str(compute_dtype)
    run_config["resolved_expert_targets"] = expert_targets
    run_config["resolved_expert_rank"] = expert_rank
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(run_config, file, ensure_ascii=False, indent=2)

    if args.save_merged_model:
        merged_dir = output_dir / "merged_model"
        LOGGER.warning(
            "Merging a 30B MoE adapter may require substantial CPU/GPU memory."
        )
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(
            merged_dir,
            safe_serialization=True,
            max_shard_size="4GB",
        )
        tokenizer.save_pretrained(merged_dir)
        LOGGER.info("Merged model saved to %s", merged_dir)

    LOGGER.info("MoE QLoRA training completed successfully.")


if __name__ == "__main__":
    main()
