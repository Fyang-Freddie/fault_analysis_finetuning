from docx import Document
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig


def extract_docx_content(docx_path: str) -> str:
    """
    提取单个 docx 文件中的文本内容，包括普通段落和表格内容。
    """
    doc = Document(docx_path)
    content_parts = []

    # 1. 提取普通段落
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            content_parts.append(text)

    # 2. 提取表格内容
    for table in doc.tables:
        for row in table.rows:
            row_text = []

            for cell in row.cells:
                cell_text = cell.text.strip()
                if cell_text:
                    row_text.append(cell_text)

            if row_text:
                content_parts.append(" | ".join(row_text))

    return "\n".join(content_parts)


FILTER_PROMPT = """
你是一名工程失效分析报告文本清洗助手。

下面是一份从 Word 文档中直接提取出来的原始文本，内容可能包含大量噪声，例如：
目录、页码、图题、表题、图片编号、图片标注、位置标签、重复标题、无意义短词等。

你的任务是：只保留对失效分析有价值的正文内容，并删除无意义噪声。

请严格遵守以下要求：

1. 删除目录内容，例如“目录”“目 录”“1 概述 1”“5.1 宏观检查 7”等。
2. 删除图题和表题，例如“图5-1 来样管外壁宏观检查”“表5-1 化学成分分析结果”等。
3. 删除图片标注和位置标签，例如“宏观”“内壁”“外壁”“中部”“位置A”“位置B”“区域1”“区域2”等单独成行的内容。
4. 删除页码、空行、重复标题、无意义编号。
5. 保留真正有分析价值的内容，包括：
   - 设备背景
   - 故障现象
   - 样品描述
   - 检测方法
   - 宏观检查结果
   - 化学成分分析结果
   - 金相组织结果
   - 硬度测试结果
   - 微观形貌 / 能谱分析结果
   - 失效机理
   - 原因分析
   - 结论与建议
6. 不要改写技术含义。
7. 不要补充原文没有的信息。
8. 不要生成问答。
9. 不要解释你的处理过程。
10. 只输出清洗后的正文文本。

原始文本如下：
{raw_text}
"""


def clean_text_with_local_qwen(raw_text: str, tokenizer, model) -> str:
    """
    使用本地 Qwen 模型清洗文本。
    """
    prompt = FILTER_PROMPT.format(raw_text=raw_text)

    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
            temperature=0.1,
            top_p=0.9
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]

    result = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    return result.strip()


def main():
    input_folder = "papers_failure_analysis2"
    model_path = r"models\Qwen3-8B"

    print("正在加载模型...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()

    print("模型加载完成，开始处理 Word 文件。")

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".docx") and not filename.startswith("~$"):
            file_path = os.path.join(input_folder, filename)

            print(f"\n正在处理：{filename}")

            raw_text = extract_docx_content(file_path)

            cleaned_text = clean_text_with_local_qwen(
                raw_text=raw_text,
                tokenizer=tokenizer,
                model=model
            )

            print("清洗后的内容：")
            print(cleaned_text)
            print("=" * 80)


if __name__ == "__main__":
    main()