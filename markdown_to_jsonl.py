#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把结构化失效分析 Markdown 转换为 QLoRA 所需的 messages JSONL。

转换规则：
1. 从文档开头到“失效机理”标题之前的全部内容作为 user 输入。
2. “失效机理”正文作为 assistant 的 <think> 分析内容。
3. “失效机理、失效结论、后续建议”同时组成 assistant 最终输出的 JSON。
4. 一篇 Markdown 生成一个训练样本。
5. 按固定随机种子划分训练集和验证集，保证重复运行结果一致。

默认用法：
python markdown_to_jsonl.py

等价于：
python markdown_to_jsonl.py \
  --input_dir data/markdown/markdown \
  --train_output data/train.jsonl \
  --validation_output data/val.jsonl \
  --validation_ratio 0.1 \
  --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

LOGGER = logging.getLogger(__name__)


# system 提示词会写入每一条训练样本。
# 明确要求“仅依据输入事实”，减少模型在失效机理和建议中补造数据。
DEFAULT_SYSTEM_PROMPT = """你是一名严谨的设备失效分析专家。
请仅依据用户提供的背景知识与检测事实完成分析，不得虚构检测数据、材料牌号、标准要求或失效现象。
先在 <think>...</think> 中给出基于证据的失效机理分析，再输出一个合法 JSON 对象。
JSON 必须且只能包含“失效机理”“失效结论”“后续建议”三个字段。"""

TARGET_SECTIONS = ("失效机理", "失效结论", "后续建议")

# 只识别二到四级 Markdown 标题，例如“## 失效结论”或“### 失效机理”。
HEADING_PATTERN = re.compile(
    r"^(?P<marks>#{2,4})[ \t]*(?P<title>.+?)[ \t]*$",
    flags=re.MULTILINE,
)


@dataclass(frozen=True)
class Heading:
    """记录一个 Markdown 标题的位置，便于准确截取章节正文。"""

    level: int
    title: str
    start: int
    end: int


@dataclass(frozen=True)
class ConvertedSample:
    """保存生成样本及来源文件名，来源只用于日志，不写入训练 JSONL。"""

    source_file: str
    messages: List[Dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert failure-analysis Markdown files to messages JSONL"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/markdown",
        help="存放 Markdown 文件的目录。",
    )
    parser.add_argument(
        "--train_output",
        type=str,
        default="data/train.jsonl",
        help="训练集 JSONL 输出路径。",
    )
    parser.add_argument(
        "--validation_output",
        type=str,
        default="data/val.jsonl",
        help="验证集 JSONL 输出路径。",
    )
    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.1,
        help="验证集比例，必须大于 0 且小于 1。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="训练集/验证集随机划分种子。",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="写入每条样本的 system 提示词。",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否递归读取 input_dir 的子目录。",
    )
    parser.add_argument(
        "--skip_invalid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="遇到缺少目标章节的文件时跳过；默认直接报错，避免静默丢数据。",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否覆盖已经存在的输出文件。",
    )
    return parser.parse_args()


def normalize_title(title: str) -> str:
    """去掉标题两侧空白和可选的中文/英文冒号，便于匹配固定章节名。"""
    return title.strip().rstrip("：:").strip()


def collect_headings(text: str) -> List[Heading]:
    """按文档顺序收集所有二到四级标题。"""
    headings: List[Heading] = []
    for match in HEADING_PATTERN.finditer(text):
        headings.append(
            Heading(
                level=len(match.group("marks")),
                title=normalize_title(match.group("title")),
                start=match.start(),
                end=match.end(),
            )
        )
    return headings


def find_heading(
    headings: List[Heading],
    title: str,
) -> Optional[Heading]:
    """查找指定章节；重复标题视为数据异常，避免截取到错误位置。"""
    matches = [heading for heading in headings if heading.title == title]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"存在重复章节标题：{title}")
    return matches[0]


def extract_section(
    text: str,
    headings: List[Heading],
    target: Heading,
) -> str:
    """
    截取目标标题下的正文。

    遇到同级或更高级标题时停止；更低级标题属于当前章节的一部分。
    例如“## 后续建议”会在下一个“## 经验标签”前结束。
    """
    section_end = len(text)
    for heading in headings:
        if heading.start <= target.start:
            continue
        if heading.level <= target.level:
            section_end = heading.start
            break
    return text[target.end:section_end].strip()


def convert_markdown(
    path: Path,
    system_prompt: str,
) -> ConvertedSample:
    """把单篇 Markdown 转成一个 system/user/assistant 对话样本。"""
    text = path.read_text(encoding="utf-8").lstrip("\ufeff").strip()
    if not text:
        raise ValueError("文件内容为空")

    headings = collect_headings(text)
    targets: Dict[str, Heading] = {}
    for section_name in TARGET_SECTIONS:
        heading = find_heading(headings, section_name)
        if heading is None:
            raise ValueError(f"缺少章节：{section_name}")
        targets[section_name] = heading

    mechanism_heading = targets["失效机理"]
    conclusion_heading = targets["失效结论"]
    advice_heading = targets["后续建议"]

    # 三个目标章节必须按“机理 → 结论 → 建议”排列，否则输入/输出边界不可信。
    if not (
        mechanism_heading.start
        < conclusion_heading.start
        < advice_heading.start
    ):
        raise ValueError("目标章节顺序必须为：失效机理、失效结论、后续建议")

    user_content = text[:mechanism_heading.start].strip()
    if not user_content:
        raise ValueError("失效机理之前没有可作为 user 输入的背景内容")

    mechanism = extract_section(text, headings, mechanism_heading)
    conclusion = extract_section(text, headings, conclusion_heading)
    advice = extract_section(text, headings, advice_heading)

    for section_name, section_content in (
        ("失效机理", mechanism),
        ("失效结论", conclusion),
        ("后续建议", advice),
    ):
        if not section_content:
            raise ValueError(f"章节正文为空：{section_name}")

    final_answer = {
        "失效机理": mechanism,
        "失效结论": conclusion,
        "后续建议": advice,
    }

    # “失效机理”本身就是基于前文检测事实形成的分析过程，因此放入 think。
    # 最终 JSON 再保留三个完整字段，便于推理阶段稳定解析输出。
    assistant_content = (
        f"<think>\n{mechanism}\n</think>\n"
        f"{json.dumps(final_answer, ensure_ascii=False, indent=2)}"
    )

    return ConvertedSample(
        source_file=path.name,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    )


def discover_markdown_files(
    input_dir: Path,
    recursive: bool,
) -> List[Path]:
    """按路径排序读取 Markdown，确保随机划分前的输入顺序稳定。"""
    pattern = "**/*.md" if recursive else "*.md"
    return sorted(
        (path for path in input_dir.glob(pattern) if path.is_file()),
        key=lambda path: str(path).casefold(),
    )


def write_jsonl(
    path: Path,
    samples: List[ConvertedSample],
) -> None:
    """以 UTF-8 JSONL 写入，每个样本严格占一行。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for sample in samples:
            json.dump(
                {"messages": sample.messages},
                file,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            file.write("\n")


def ensure_output_paths(
    train_output: Path,
    validation_output: Path,
    overwrite: bool,
) -> None:
    """防止误覆盖已有训练数据，也禁止训练集和验证集写到同一文件。"""
    if train_output.resolve() == validation_output.resolve():
        raise ValueError("训练集与验证集输出路径不能相同。")

    existing = [
        path
        for path in (train_output, validation_output)
        if path.exists()
    ]
    if existing and not overwrite:
        paths = "、".join(str(path) for path in existing)
        raise FileExistsError(
            f"输出文件已存在：{paths}。如确认覆盖，请传入 --overwrite。"
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("--validation_ratio 必须大于 0 且小于 1。")
    if not args.system_prompt.strip():
        raise ValueError("--system_prompt 不能为空。")

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Markdown 目录不存在：{input_dir}")

    train_output = Path(args.train_output)
    validation_output = Path(args.validation_output)
    ensure_output_paths(
        train_output=train_output,
        validation_output=validation_output,
        overwrite=args.overwrite,
    )

    markdown_files = discover_markdown_files(
        input_dir=input_dir,
        recursive=args.recursive,
    )
    if not markdown_files:
        raise FileNotFoundError(f"{input_dir} 中没有找到 Markdown 文件。")

    converted: List[ConvertedSample] = []
    rejected: List[str] = []
    for path in markdown_files:
        try:
            converted.append(
                convert_markdown(
                    path=path,
                    system_prompt=args.system_prompt,
                )
            )
        except (OSError, UnicodeError, ValueError) as exc:
            message = f"{path}: {exc}"
            if not args.skip_invalid:
                raise ValueError(message) from exc
            rejected.append(message)
            LOGGER.warning("Skipped invalid Markdown: %s", message)

    if len(converted) < 2:
        raise RuntimeError("有效样本少于 2 条，无法划分训练集和验证集。")

    # 在排序后的样本上使用固定随机种子洗牌，确保相同输入和 seed 得到相同划分。
    random.Random(args.seed).shuffle(converted)
    validation_count = max(
        1,
        round(len(converted) * args.validation_ratio),
    )
    validation_count = min(validation_count, len(converted) - 1)

    validation_samples = converted[:validation_count]
    train_samples = converted[validation_count:]

    write_jsonl(train_output, train_samples)
    write_jsonl(validation_output, validation_samples)

    LOGGER.info(
        "Conversion completed: total=%d, train=%d, validation=%d, rejected=%d",
        len(converted),
        len(train_samples),
        len(validation_samples),
        len(rejected),
    )
    LOGGER.info("Train JSONL: %s", train_output.resolve())
    LOGGER.info("Validation JSONL: %s", validation_output.resolve())


if __name__ == "__main__":
    main()
