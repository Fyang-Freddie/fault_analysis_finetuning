from docx import Document
import argparse
import re
import os
import json
from openai import OpenAI
from dotenv import load_dotenv


# =========================
# 1. 读取 .env 配置
# =========================
load_dotenv()

GLM_API_KEY = os.getenv("API_KEY")
GLM_BASE_URL = os.getenv("API_BASE_URL")

client = OpenAI(
    api_key=GLM_API_KEY,
    base_url=GLM_BASE_URL
)


# =========================
# 2. 这里填写你自己的提示词
# =========================
GENERATION_PROMPT ="""
你是一名工业设备失效分析专家，熟悉核电、火电、石化、钢铁、船舶、矿山等场景中的通用设备失效分析，包括管道、阀门、水泵、风机、换热器、压力容器、紧固件、轴承、焊接接头、密封件等设备或部件。

你的任务是根据输入的技术报告文本，抽取高质量中文问答数据，用于训练“工业设备失效分析问答模型”。

请注意：你要生成的是“基于失效事实和检测证据进行工程判断”的问答数据，而不是阅读理解题、报告总结题、检测流程管理题或泛泛的方法论问题。

一、核心生成目标

你生成的每一条 QA 都必须围绕以下逻辑展开：

发现了什么事实情况 → 这些事实说明什么工程问题 → 可以支持什么失效结论 → 应提出什么工程建议

也就是说，问题和答案应重点训练模型完成以下能力：

1. 从故障现象中识别异常特征。
2. 从宏观检查、化学成分、金相检验、硬度试验、微观分析中提取关键证据。
3. 根据多项检测证据判断可能的失效机理。
4. 根据证据排除不符合的失效原因。
5. 在证据充分时给出失效结论。
6. 在证据不足时说明不能直接下结论，并提出进一步检测建议。
7. 根据失效机理提出工程整改、预防或运行维护建议。

二、必须生成的问题类型

请优先生成以下类型的问题：

1. 事实发现 → 机理判断型

问题应围绕“检测发现了什么异常现象，这些异常现象说明可能发生了什么失效机理”。

示例方向：
“某承压管道内壁发现点蚀坑，裂纹从内壁起裂并向外壁扩展，裂纹呈分叉、尖细和树枝状特征，同时材料成分和基体组织未见明显异常，应如何判断其主要失效机理？”

2. 证据组合 → 结论推导型

问题应要求模型综合多个检测结果，而不是只解释单个现象。

示例方向：
“当宏观检查显示裂纹起源于介质接触表面，金相检验显示裂纹与腐蚀坑相连，硬度和化学成分均符合要求时，应如何综合判断失效原因？”

3. 异常现象 → 风险判断型

问题应围绕某些检测事实说明设备存在什么风险。

示例方向：
“如果某设备部件表面存在较多腐蚀坑，且裂纹多从腐蚀坑处萌生，这对后续运行安全意味着什么？”

4. 检测证据 → 排除原因型

问题应体现如何根据证据排除材料错用、热处理异常、单纯过载、制造缺陷等原因。

示例方向：
“如果失效件化学成分符合标准、硬度无明显异常、基体组织正常，但裂纹均从内壁腐蚀坑处起裂，可以优先排除哪些失效原因？”

5. 结论 → 工程建议型

问题应围绕失效分析结论提出针对性建议。

示例方向：
“当多项检测结果支持腐蚀诱发开裂时，应从介质控制、表面防护、定期检测和运行工况管理等方面提出哪些工程建议？”

6. 证据不足 → 补充检测型

如果输入文本证据不足，不要强行生成失效机理结论，应生成进一步检测建议型问题。

示例方向：
“如果现有资料只显示部件存在裂纹和腐蚀痕迹，但缺少金相、硬度和微观断口分析，应如何安排后续检测以判断失效机理？”

三、禁止生成的问题类型

不得生成以下类型的问题：

1. 检测流程管理类问题

例如不得生成：
“多件相似失效样品分析中，如何处理以保证效率与代表性？”
“如何安排多个样品的检测顺序？”
“如何提高失效分析工作的效率？”

2. 报告写作类问题

例如不得生成：
“失效分析报告应如何组织章节？”
“报告中图表应该如何编号？”
“如何描述多个样品的相似性？”

3. 泛泛方法论问题

例如不得生成：
“失效分析通常包括哪些步骤？”
“宏观检查有什么作用？”
“金相检验为什么重要？”

除非输入证据不足，只能生成补充检测建议型 QA，否则不要生成这种泛泛的知识解释题。

4. 单纯阅读理解题

例如不得生成：
“该报告检测了哪些样品？”
“该文件中的表 5-1 说明了什么？”
“该报告第几章给出了结论？”
“某编号样品的检测结果是什么？”

5. 与失效结论无直接关系的问题

例如不得生成：
“多个样品形貌相似时如何提高分析代表性？”
“相似样品是否可以合并描述？”
“检测报告如何避免重复表述？”

四、问题生成范围

生成的问题必须围绕以下五类检测证据展开，并尽量与失效结论或工程建议关联。

1. 宏观检查

关注：
裂纹位置、起裂区域、断口形貌、腐蚀坑、磨损痕迹、变形、泄漏位置、表面颜色、沉积物、焊缝附近异常等。

问题应体现：
这些宏观事实提示了什么失效方向，例如腐蚀、疲劳、磨损、过载、制造缺陷或介质作用等。

2. 化学成分分析

关注：
材料成分是否符合标准、关键合金元素是否异常、是否存在材料错用、元素偏析、腐蚀介质残留等。

问题应体现：
化学成分结果如何支持或排除材料错用、材料不合格、耐蚀性不足等原因。

3. 金相检验

关注：
显微组织类型、夹杂物、晶粒状态、脱碳、过热、组织异常、裂纹扩展路径、晶间或穿晶特征、腐蚀坑与裂纹关系等。

问题应体现：
金相结果如何判断裂纹起源、扩展方式和可能失效机理。

4. 硬度试验

关注：
硬度是否异常、硬度分布是否均匀、是否存在局部硬化或软化、热处理状态是否异常、硬度与开裂、磨损、脆化之间的关系等。

问题应体现：
硬度结果如何支持或排除热处理异常、材料强度异常、加工硬化、局部脆化等原因。

5. 微观分析

关注：
SEM 断口形貌、韧窝、解理、疲劳条带、沿晶断裂、腐蚀产物、能谱分析、沉积物成分、氧化物、氯、硫等腐蚀性元素。

问题应体现：
微观形貌和微区成分如何支撑疲劳、脆断、腐蚀、磨损、氧化或介质作用等机理判断。

五、证据使用规则

1. 所有问答都必须基于输入文本中的证据。
2. 不得虚构输入中没有出现的材料牌号、温度、压力、介质、检测数据、设备编号、标准编号或实验结果。
3. 可以对具体设备、样品和编号进行抽象，但不能改变事实含义。
4. 如果输入中有具体编号，应抽象为“某工业设备部件”“某承压管道”“某换热器管束”“某阀门部件”“某焊接接头”等。
5. 如果输入中出现具体数值，可以在 answer 中概括其工程意义，但不要把 question 设计成单纯询问数值。
6. 如果证据不足，不得生成确定性失效结论，只能生成证据解释型、现象描述型或进一步检测建议型 QA。
7. thinking 中必须体现证据链，而不是简单重复答案。

六、抽象化要求

生成问题时必须进行工程抽象，避免针对原文细节发问。

请避免以下写法：

* “该报告中的 001VP 管失效原因是什么？”
* “图 5-1 显示了什么？”
* “表 5-1 的化学成分结果是什么？”
* “该文件第 3 章的结论是什么？”
* “本文中的样品 A 的硬度是多少？”

应改写为通用工程失效分析问题，例如：

* “某承压管道内壁存在点蚀坑，裂纹从点蚀坑处起裂并呈分叉扩展时，应如何判断其失效机理？”
* “当材料化学成分满足标准要求但仍发生裂纹泄漏时，应如何结合宏观和金相结果继续排查原因？”
* “如果硬度和组织均无明显异常，而裂纹起源于介质接触表面的腐蚀坑，应如何判断材料因素与环境因素在失效中的作用？”

七、输出数量要求

请根据输入文本的信息密度生成问答对。

* 如果输入内容较少，生成 3–5 条高质量 QA。
* 如果输入内容较完整，生成 5–10 条高质量 QA。
* 如果输入包含完整的宏观、化学成分、金相、硬度和微观分析内容，可以生成 10–15 条 QA。

质量优先，不要为了凑数量生成重复、空泛或证据不足的问题。

八、输出格式要求

你必须只输出合法 JSONL，不要输出 Markdown，不要输出 ```json 代码块，不要输出解释性文字。

每一行必须是一个独立 JSON 对象，不要输出 JSON 数组。

每个 JSON 对象格式如下：

{
"question": "面向通用工业设备失效分析场景的中文问题",
"thinking": [
"1. 发现的关键故障事实、检测结果或运行异常。",
"2. 该事实说明的工程含义，例如起裂位置、损伤类型、材料状态或环境作用。",
"3. 另一项关键检测证据及其工程含义。",
"4. 多项证据之间如何相互印证，并指向某类失效机理。",
"5. 根据现有证据可以排除或暂不能确认的其他原因。",
"6. 最终可支持的失效结论、风险判断、工程建议或进一步检测建议。"
],
"answer": "基于事实和证据得到的简洁中文结论或建议。"
}

九、字段要求

1. question

* 必须是中文。
* 必须围绕“事实情况 → 失效结论 → 工程建议”展开。
* 必须是通用工程设备失效分析问题。
* 不要出现具体文件名、图号、表号、样品编号、页码。
* 不要生成检测流程管理、报告写作、样品代表性或效率优化类问题。
* 尽量体现推理性，不要只问“是什么”。

2. thinking

* 必须是数组。
* 每条 thinking 必须包含证据和推理含义。
* 应体现“现象—证据—分析—排除—结论/建议”的过程。
* 不要编造输入中不存在的证据。
* 如果证据不足，应明确说明“现有证据不足以直接判断最终机理”。

3. answer

* 必须是中文。
* 应简洁、明确、工程化。
* 应回答“发现了什么事实、可以支持什么判断、应采取什么建议”。
* 不要写成泛泛的教材解释。
* 不要脱离输入内容泛泛而谈。

十、生成原则

请优先生成能够训练模型完成以下任务的 QA：

1. 根据宏观裂纹、腐蚀坑、断口、泄漏位置判断可能失效方向。
2. 根据化学成分分析判断材料是否符合要求，以及是否可以排除材料错用。
3. 根据金相组织和裂纹扩展路径判断裂纹起源和扩展方式。
4. 根据硬度结果判断材料状态、加工影响或热处理异常。
5. 根据微观断口和能谱结果判断疲劳、脆断、腐蚀、磨损、氧化或介质作用。
6. 综合多种检测结果判断失效机理。
7. 根据失效机理提出工程预防和整改建议。
8. 在证据不足时提出合理的补充检测方案。

十一、最终检查

输出前请逐条检查：

1. 这个问题是否围绕具体失效事实和检测证据？
2. 这个问题是否能训练模型判断失效原因、风险或工程建议？
3. 这个问题是否避免了报告写作、样品管理、流程效率、代表性分析等无关方向？
4. 这个问题是否避免了针对具体文件、图表、编号的阅读理解？
5. answer 是否是由输入证据支持的结论或建议？

只有同时满足以上要求的 QA 才能输出。
"""


def is_noise_line(text: str) -> bool:
    """
    判断一行文本是否属于噪声内容，例如：
    目录行、图题、表题、图片标注、页码、小标签等
    """

    text = text.strip()

    if not text:
        return True

    # 1. 过滤很短的图片标注词
    short_noise_words = {
        "宏观", "内壁", "外壁", "中部", "整体情况",
        "位置A", "位置B", "位置C", "位置D", "位置E", "位置F",
        "启裂处", "裂纹尖端", "中间位置",
        "001VP", "002VP",
        "区域1", "区域2", "区域3", "区域4"
    }

    if text in short_noise_words:
        return True

    # 2. 过滤图题，例如：图5-1 来样管外壁宏观检查
    if re.match(r"^图\s*\d+[-－]\d+", text):
        return True

    # 3. 过滤表题，例如：表5-1 化学成分分析结果
    if re.match(r"^表\s*\d+[-－]\d+", text):
        return True

    # 4. 过滤目录标题
    if text in {"目录", "目  录", "目 录"}:
        return True

    # 5. 过滤目录中的行，例如：5.1 宏观检查    7
    if re.match(r"^\d+(\.\d+)*\s+.+\s+\d+$", text):
        return True

    # 6. 过滤纯页码
    if re.match(r"^\d+$", text):
        return True

    # 7. 过滤类似：(a)、(b)、（a）、（b）
    if re.match(r"^[\(（][a-zA-Z0-9一二三四五六七八九十]+[\)）]", text):
        return True

    # 8. 过滤只有编号或位置编号的行，例如：位置1、位置10
    if re.match(r"^位置\d+$", text):
        return True

    # 9. 过滤过短且没有中文说明意义的行
    if len(text) <= 2:
        return True

    return False


def clean_text_lines(lines: list) -> list:
    """
    对提取出来的文本行进行清洗。
    """
    cleaned = []

    for line in lines:
        line = line.strip()

        if is_noise_line(line):
            continue

        # 合并多余空格
        line = re.sub(r"\s+", " ", line)

        cleaned.append(line)

    return cleaned


def extract_docx_content(docx_path: str, extract_tables: bool = True) -> str:
    """
    提取并清洗单个 docx 文件中的文本内容。

    参数:
        docx_path: docx 文件路径
        extract_tables: 是否提取表格内容，默认提取

    返回:
        str: 清洗后的完整文本内容
    """

    doc = Document(docx_path)
    content_parts = []

    # 1. 提取普通段落
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            content_parts.append(text)

    # 2. 提取表格内容
    if extract_tables:
        for table in doc.tables:
            for row in table.rows:
                row_text = []

                for cell in row.cells:
                    cell_text = cell.text.strip()

                    if cell_text and not is_noise_line(cell_text):
                        row_text.append(cell_text)

                if row_text:
                    content_parts.append(" | ".join(row_text))

    # 3. 清洗噪声行
    cleaned_lines = clean_text_lines(content_parts)

    return "\n".join(cleaned_lines)


def call_glm_generate(cleaned_text: str) -> str:
    """
    将清洗后的文本输入 GLM 模型，生成数据。

    参数:
        cleaned_text: 单篇 docx 清洗后的文本

    返回:
        str: GLM 模型生成结果
    """

    user_content = f"""
{GENERATION_PROMPT}

以下是清洗后的原始文本：

{cleaned_text}
"""

    resp = client.chat.completions.create(
        model="glm-latest",
        messages=[
            {"role": "user", "content": user_content}
        ],
        temperature=0
    )

    return resp.choices[0].message.content


def get_last_jsonl_record(output_path: str):
    """
    读取 jsonl 文件最后一条非空 JSON 数据，用于断点续写。
    """

    if not os.path.exists(output_path):
        return None

    with open(output_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in reversed(lines):
        line = line.strip()
        if line:
            return json.loads(line)

    return None


def save_to_jsonl(output_path: str, generated_data: str, source_pdf: str, qa_type: str):
    """
    将模型生成的 JSONL 数据补充固定字段后追加写入 jsonl 文件。
    """

    saved_count = 0
    with open(output_path, "a", encoding="utf-8") as f:
        for line in generated_data.strip().splitlines():
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            item["qa_type"] = qa_type
            item["source_pdf"] = source_pdf
            item["page_start"] = 0
            item["page_end"] = 0
            item["chunk_id"] = 1

            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            saved_count += 1

    return saved_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", default="papers_failure_analysis2")
    parser.add_argument("--output_file", default="qa_word.jsonl")
    parser.add_argument("--error_file", default="error_log.txt")
    parser.add_argument("--qa_type", default="")
    parser.add_argument("--write_mode", choices=["resume", "rewrite"], default="resume")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_file = args.output_file
    error_file = args.error_file

    if args.write_mode == "rewrite":
        open(output_file, "w", encoding="utf-8").close()

    filenames = [
        filename for filename in os.listdir(input_folder)
        if filename.lower().endswith(".docx") and not filename.startswith("~$")
    ]

    if args.write_mode == "resume":
        last_record = get_last_jsonl_record(output_file)
        if last_record:
            last_source_pdf = last_record.get("source_pdf")
            if last_source_pdf in filenames:
                last_index = filenames.index(last_source_pdf)
                filenames = filenames[last_index + 1:]
                print(f"断点续写：从 {last_source_pdf} 之后继续处理")
            else:
                print("断点续写：最后一条数据未匹配到当前文件夹中的文件，将从头开始处理")

    for filename in filenames:
        file_path = os.path.join(input_folder, filename)

        try:
            print(f"正在处理：{filename}")

            # 1. 先清洗当前文件
            text = extract_docx_content(file_path, extract_tables=False)

            if not text.strip():
                print(f"{filename} 清洗后文本为空，跳过")
                continue

            print("文本清洗完成，正在调用 GLM 模型生成数据...")

            # 2. 当前文件清洗完成后，立即调用模型生成
            generated_data = call_glm_generate(text)

            print("模型生成完成，正在写入 qa_word.jsonl...")

            # 3. 补充固定字段后写入 JSONL 文件
            saved_count = save_to_jsonl(
                output_path=output_file,
                generated_data=generated_data,
                source_pdf=filename,
                qa_type=args.qa_type
            )

            print(f"{filename} 已保存完成，写入 {saved_count} 条数据")
            print("=" * 80)

        except Exception as e:
            print(f"{filename} 处理失败：{e}")

            with open(error_file, "a", encoding="utf-8") as f:
                f.write(f"{filename}\t{str(e)}\n")

            print("=" * 80)

if __name__ == "__main__":
    main()
