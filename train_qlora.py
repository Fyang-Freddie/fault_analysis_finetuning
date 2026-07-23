#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用纯文本因果语言模型 QLoRA 训练脚本

数据格式（JSONL，每行一个样本）：
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "背景知识与检测事实..."},
    {"role": "assistant", "content": "<think>...</think>\n{...}"}
  ]
}

特点：
1. 4-bit NF4 QLoRA。
2. 只对 assistant 回复计算 loss。
3. 支持断点续训、验证集、最佳模型保存。
4. 默认参数针对约 300 条高质量数据。

使用 Qwen3 8B：
python train_qlora.py  --model_name_or_path models/Qwen3-8B --train_file data/train.jsonl --validation_file data/val.jsonl --output_dir outputs/qwen3_8b_failure_qlora

使用更大的纯文本 Qwen 模型时，只需替换模型名和输出目录：
python train_qlora.py \
  --model_name_or_path Qwen/Qwen3-32B \
  --output_dir outputs/qwen3_32b_failure_qlora

注意：Qwen/Qwen3.5-35B-A3B 是带视觉编码器的多模态模型，不能直接套用
本脚本的 AutoModelForCausalLM + AutoTokenizer 纯文本路径。

多卡数据并行（每张卡都会加载一份 4-bit 模型）：
accelerate launch --multi_gpu train_qlora.py ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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

LOGGER = logging.getLogger(__name__)

# markdown_to_jsonl.py 生成的 assistant 最终 JSON 必须只包含这三个字段。
EXPECTED_TARGET_KEYS = {"失效机理", "失效结论", "后续建议"}
ASSISTANT_TARGET_PATTERN = re.compile(
    r"^<think>\s*(?P<thinking>.*?)\s*</think>\s*(?P<answer>\{.*\})\s*$",
    flags=re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text causal-LM failure-analysis QLoRA")

    # 默认训练 Qwen3-8B；运行时仍可通过该参数切换为本地模型或其他兼容模型。
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-8B",
        help=(
            "Hugging Face 纯文本因果语言模型名称或本地目录，例如 "
            "Qwen/Qwen3-8B、Qwen/Qwen3-32B。Qwen3 需要 transformers>=4.51.0。"
        ),
    )
    parser.add_argument(
        "--model_revision",
        type=str,
        default=None,
        help="可选：模型分支、标签或 commit ID；正式训练建议固定 commit ID 以便复现。",
    )
    parser.add_argument("--train_file", type=str, default="data/train.jsonl")
    parser.add_argument("--validation_file", type=str, default="data/val.jsonl")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/qlora_failure_analysis",
    )

    # 当前 data/train.jsonl 的最长样本明显长于普通短对话，默认提高到 4096。
    # 实际 token 数由 tokenizer 决定；超长样本默认报错，不会静默截断目标答案。
    parser.add_argument("--max_length", type=int, default=4096)
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
        default="all-linear",
        help="推荐 all-linear；也可传逗号分隔的模块名称。",
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
            "是否执行模型仓库中的自定义 Python 代码。为安全起见默认关闭；"
            "仅对可信仓库显式传入 --trust_remote_code。"
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
        help="默认关闭，只训练 assistant 输出。",
    )
    parser.add_argument(
        "--enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "传给 Qwen3 chat template 的 thinking 开关。当前训练数据含 <think>，"
            "因此默认开启；应与推理阶段保持一致。"
        ),
    )
    parser.add_argument(
        "--validate_target_format",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "训练前校验 assistant 是否为 <think>...</think> 加三个固定 JSON 字段。"
        ),
    )
    parser.add_argument(
        "--allow_truncated_samples",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "是否允许截断超过 max_length 的样本。默认禁止，"
            "避免从右侧截掉失效结论或后续建议。"
        ),
    )
    parser.add_argument(
        "--save_merged_model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="训练结束后尝试合并 LoRA；大模型会占用大量 CPU/GPU 内存，默认不合并。",
    )

    return parser.parse_args()


def validate_assistant_target(
    content: str,
    file_path: Path,
    line_number: int,
) -> None:
    """校验转换脚本约定的 think + JSON 输出格式。"""
    match = ASSISTANT_TARGET_PATTERN.fullmatch(content.strip())
    if match is None:
        raise ValueError(
            f"{file_path} 第 {line_number} 行的 assistant 输出必须是 "
            "<think>...</think> 后跟 JSON 对象。"
        )
    if not match.group("thinking").strip():
        raise ValueError(
            f"{file_path} 第 {line_number} 行的 <think> 内容不能为空。"
        )

    try:
        answer = json.loads(match.group("answer"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{file_path} 第 {line_number} 行的 assistant 最终答案不是合法 JSON：{exc}"
        ) from exc

    if not isinstance(answer, dict) or set(answer) != EXPECTED_TARGET_KEYS:
        raise ValueError(
            f"{file_path} 第 {line_number} 行的 assistant JSON 必须且只能包含："
            f"{'、'.join(sorted(EXPECTED_TARGET_KEYS))}。"
        )
    for key, value in answer.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{file_path} 第 {line_number} 行的 assistant JSON 字段 "
                f"{key!r} 必须是非空字符串。"
            )


def validate_jsonl(
    path: str,
    validate_target_format: bool = False,
) -> None:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"数据文件不存在：{file_path}")

    valid_count = 0
    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{file_path} 第 {line_number} 行不是合法JSON：{exc}"
                ) from exc

            if not isinstance(sample, dict):
                raise ValueError(
                    f"{file_path} 第 {line_number} 行的顶层 JSON 必须是对象。"
                )

            messages = sample.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(
                    f"{file_path} 第 {line_number} 行缺少非空 messages 列表。"
                )

            assistant_messages = [
                message
                for message in messages
                if isinstance(message, dict) and message.get("role") == "assistant"
            ]
            if not assistant_messages:
                raise ValueError(
                    f"{file_path} 第 {line_number} 行没有 assistant 回复。"
                )

            # chat template 通常要求每条消息都含有字符串 role/content；
            # 在这里提前报出具体行号，比数据 map 阶段的模板异常更容易定位。
            for message_index, message in enumerate(messages):
                if not isinstance(message, dict):
                    raise ValueError(
                        f"{file_path} 第 {line_number} 行的第 {message_index + 1} 条消息不是对象。"
                    )
                if not isinstance(message.get("role"), str) or not isinstance(
                    message.get("content"), str
                ):
                    raise ValueError(
                        f"{file_path} 第 {line_number} 行的第 {message_index + 1} 条消息"
                        "必须包含字符串 role 和 content。"
                    )

            if validate_target_format:
                validate_assistant_target(
                    content=assistant_messages[-1]["content"],
                    file_path=file_path,
                    line_number=line_number,
                )
            valid_count += 1

    if valid_count == 0:
        raise ValueError(f"{file_path} 没有有效样本。")

    LOGGER.info("Validated %s: %d samples", file_path, valid_count)


def find_last_assistant_index(messages: List[Dict[str, Any]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return index
    raise ValueError("样本没有 assistant 消息。")


def build_preprocess_function(
    tokenizer: AutoTokenizer,
    max_length: int,
    train_on_inputs: bool,
    enable_thinking: bool = True,
    # 默认 True 用于兼容复用该函数的旧调用方；train_qlora.py 会显式传入 False。
    allow_truncated_samples: bool = True,
):
    """创建数据预处理函数。

    默认只对最后一条 assistant 回复计算 loss：
      labels = [-100, ..., -100, assistant_token_1, ...]

    通过同一 chat template 分别渲染：
      1. assistant 之前的消息 + generation prompt
      2. 完整对话
    从而得到 assistant 回复的起始 token 位置。

    Qwen3 的 chat template 默认开启 thinking 模式。如果数据中保留推理内容，
    assistant.content 应统一使用 <think>...</think> + 最终答案的格式；
    如果不希望模型输出推理过程，应在制作训练数据时统一移除。
    """

    def preprocess(example: Dict[str, Any]) -> Dict[str, List[int]]:
        messages = example["messages"]
        if not isinstance(messages, list):
            raise ValueError("messages 必须是列表。")

        assistant_index = find_last_assistant_index(messages)
        complete_messages = messages[: assistant_index + 1]

        full_text = tokenizer.apply_chat_template(
            complete_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
        full_encoding = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=False,
        )

        full_input_ids = full_encoding["input_ids"]
        full_attention_mask = full_encoding["attention_mask"]
        if len(full_input_ids) > max_length and not allow_truncated_samples:
            raise ValueError(
                f"样本分词后长度为 {len(full_input_ids)}，超过 max_length={max_length}。"
                "为避免截掉 assistant 的失效结论或后续建议，训练已停止；"
                "请增大 --max_length，或明确传入 --allow_truncated_samples。"
            )

        # tokenizer 默认从右侧截断；这会优先损失 assistant 输出，因此只在用户
        # 明确允许时执行。正常情况下上面的长度检查会保证这里无需截断。
        input_ids = full_input_ids[:max_length]
        attention_mask = full_attention_mask[:max_length]

        if train_on_inputs:
            labels = input_ids.copy()
        else:
            prompt_messages = complete_messages[:assistant_index]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )["input_ids"]

            # 某些模型的chat template在完整消息与generation prompt之间
            # 可能出现极小差异，因此取最长公共前缀，比直接使用len(prompt_ids)更稳健。
            common_prefix_length = 0
            for prompt_token, full_token in zip(prompt_ids, input_ids):
                if prompt_token != full_token:
                    break
                common_prefix_length += 1

            if common_prefix_length == 0:
                raise ValueError(
                    "无法定位 assistant 起始位置，请检查模型chat template和数据格式。"
                )

            labels = [-100] * common_prefix_length + input_ids[common_prefix_length:]

        # 如果截断后 assistant 内容完全被切掉，该样本无法训练。
        if all(label == -100 for label in labels):
            raise ValueError(
                "样本在max_length截断后没有保留assistant输出。"
                "请缩短输入背景或增大--max_length。"
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return preprocess


@dataclass
class AssistantOnlyDataCollator:
    """对 input_ids、attention_mask、labels 进行动态padding。"""

    tokenizer: AutoTokenizer
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_length = ((max_length + multiple - 1) // multiple) * multiple

        input_ids_batch: List[List[int]] = []
        attention_mask_batch: List[List[int]] = []
        labels_batch: List[List[int]] = []

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id 未设置。")

        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            labels = feature["labels"]
            padding_length = max_length - len(input_ids)

            if self.tokenizer.padding_side == "left":
                input_ids = [pad_token_id] * padding_length + input_ids
                attention_mask = [0] * padding_length + attention_mask
                labels = [-100] * padding_length + labels
            else:
                input_ids = input_ids + [pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
                labels = labels + [-100] * padding_length

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            labels_batch.append(labels)

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }


def resolve_device_map() -> Dict[str, int]:
    """单进程或torchrun/accelerate数据并行时，将整份模型放到当前进程GPU。"""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return {"": local_rank}


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("该脚本需要NVIDIA GPU和CUDA环境。")
    if not torch.cuda.is_bf16_supported():
        LOGGER.warning("当前GPU可能不原生支持BF16，将改用FP16计算。")

    # data/train.jsonl 和 data/val.jsonl 由 markdown_to_jsonl.py 生成时，
    # 可在下载/加载大模型之前完成严格格式检查。
    validate_jsonl(
        args.train_file,
        validate_target_format=args.validate_target_format,
    )
    validate_jsonl(
        args.validation_file,
        validate_target_format=args.validate_target_format,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading tokenizer: %s", args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    # 本脚本依赖模型自带的对话模板来定位 assistant 的训练 token。
    # Base 模型或某些第三方模型可能没有 chat_template，此时不应猜测模板。
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "当前 tokenizer 没有 chat_template。请改用 Instruct/Chat 模型，"
            "或先为 tokenizer 配置与该模型匹配的对话模板。"
        )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer同时缺少pad_token_id与eos_token_id。")
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

    LOGGER.info("Loading 4-bit model. This can take several minutes.")
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

    model.config.use_cache = False
    if hasattr(model.config, "pretraining_tp"):
        model.config.pretraining_tp = 1

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )

    if args.target_modules.strip() == "all-linear":
        target_modules: str | List[str] = "all-linear"
    else:
        target_modules = [
            module.strip()
            for module in args.target_modules.split(",")
            if module.strip()
        ]
        if not target_modules:
            raise ValueError("--target_modules 不能为空。")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

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
        enable_thinking=args.enable_thinking,
        allow_truncated_samples=args.allow_truncated_samples,
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

    data_collator = AssistantOnlyDataCollator(tokenizer=tokenizer)

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
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    LOGGER.info("Starting training")
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
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(run_config, file, ensure_ascii=False, indent=2)

    if args.save_merged_model:
        merged_dir = output_dir / "merged_model"
        LOGGER.warning(
            "Merging an adapter may require enough CPU/GPU memory for the full model."
        )
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(
            merged_dir,
            safe_serialization=True,
            max_shard_size="4GB",
        )
        tokenizer.save_pretrained(merged_dir)
        LOGGER.info("Merged model saved to %s", merged_dir)

    LOGGER.info("Training completed successfully.")


if __name__ == "__main__":
    main()
