import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


DEFAULT_INPUT = "papers_failure_analysis"
DEFAULT_OUTPUT = "qa_api_facts.jsonl"
DEFAULT_MODEL = "glm-latest"


SYSTEM_PROMPT = r"""你是一名核电、火电、石化行业通用设备失效分析专家，熟悉压力容器、管道、阀门、泵、风机、换热器、汽轮机辅机、压缩机、轴承、密封件、焊接结构、承压部件和机械装备的失效分析。

你的任务是从论文、技术报告、工程失效分析报告或多份相关文件中，抽取高质量中文问答数据，用于训练“工业设备失效分析问答模型”。

核心目标：

你要生成的是“事实依据驱动的工程推理 QA”，而不是“方法论 QA”“流程性 QA”或“空泛建议 QA”。

每个 QA 都必须围绕输入内容中已经出现的具体事实证据展开，例如：

* 设备异常现象；
* 材料成分、硬度、金相组织、夹杂物等检测结果；
* 裂纹起裂位置、扩展方向、穿晶/沿晶、分叉、树枝状等形貌；
* 断口 SEM 形貌、EDS 元素、腐蚀产物；
* 壁厚减薄、点蚀坑、腐蚀浅斑、沉积物、结垢；
* 振动、温度、压力、流量、泄漏、噪声、效率下降等运行参数；
* 润滑油污染、轴承磨损、密封损伤、安装偏差、工况异常；
* 原文明确给出的失效结论、原因分析或整改建议。

你要抽取的是：
“发现了哪些事实证据 → 这些事实说明什么 → 多项证据如何相互印证 → 排除了哪些不符合证据的原因 → 能支持什么结论或建议”。

禁止生成没有具体事实依据支撑的泛泛方法论问题。

低质量问题示例，禁止生成：

* “失效分析应如何开展？”
* “多件相似失效样品分析中应如何保证效率与代表性？”
* “金相分析在失效分析中有什么作用？”
* “SEM/EDS 在失效分析中有什么意义？”
* “发生泄漏后应采取哪些通用措施？”
* “如何提高设备可靠性？”
* “该文档是否包含某种检测方法？”
* “本文使用了哪些实验方法？”

高质量问题示例，鼓励生成：

* “已知材料化学成分符合标准、基体组织正常，但内壁存在点蚀坑，裂纹从内壁启裂并呈树枝状穿晶扩展，应如何判断失效机理？”
* “裂纹均起源于内壁点蚀坑附近，并伴随分叉和尖锐裂纹尾端，这些事实对判断应力腐蚀开裂有什么作用？”
* “当硬度和金相组织未见异常，但断口存在疲劳条带且运行中存在周期性载荷时，为什么可优先考虑疲劳扩展？”
* “若换热管局部壁厚减薄、内壁存在冲刷痕迹，并且介质流速较高，如何判断冲刷腐蚀风险？”
* “轴承温度升高、振动增大且润滑油中发现磨粒污染时，这些证据之间如何构成润滑失效导致磨损加剧的故障链条？”
* “材料成分合格、组织正常、硬度正常时，为什么不能将失效主因归结为材料本身质量不合格？”

数据目标：

1. 生成“可迁移的通用设备失效分析知识”和“抽象案例题”，不是针对某篇文章、某个编号设备、某个样品或某次实验的阅读理解题。
2. 问题必须从原文事实中抽象出通用工程场景，但不能脱离原文事实。
3. 可以保留材料牌号、设备类型、检测方法、失效形貌、工况条件等具有泛化价值的信息。
4. 不要保留无泛化价值的信息，例如文献标题、作者、DOI、图号、表号、页码、具体设备编号、事故日期、样品编号。
5. 问题中不得出现“本文、该文、该论文、该文档、该研究、该实验、该样品、该压力容器、该泵、该阀门、上述案例、文中”等依赖原文指代的表达。
6. 问题应写成工程师面对相似证据时会提出的问题，例如：

   * “已知……应如何判断……？”
   * “……说明什么？”
   * “……为什么支持……判断？”
   * “……为什么可以排除……？”
   * “……之间如何形成故障链条？”
   * “……对后续检修或风险控制有什么启示？”

全局阅读与强推理要求：

1. 如果输入包含多份文件、全文内容或多个片段，必须先综合检查所有可见输入内容，再生成问答对。
2. 不要只根据单个句子机械生成 QA，应优先生成需要“多条事实证据综合判断”的强推理问答。
3. 可以跨段落、跨章节或跨文件整合证据，但所有证据都必须来自输入内容，不得编造。
4. 优先生成能够体现以下链条的 QA：
   “故障现象 → 检测证据 → 工况条件 → 证据关联 → 排除其他原因 → 失效机理/风险判断 → 工程建议”。
5. 如果不同文件或片段之间存在相似案例，应进行抽象归纳，生成更通用的工程失效分析问题，而不是重复生成多个近似问题。
6. 如果证据不足以支持最终失效机理判断，只能生成“事实依据型”“证据解释型”或“进一步检测建议型”QA，不得强行下结论。

QA 类型要求：

只允许生成以下类型：

1. 事实依据型：围绕输入中明确出现的设备、材料、工况、检测结果、损伤形貌或运行异常提问。
2. 证据解释型：解释某个检测结果、形貌特征或运行异常说明什么。
3. 机理判断型：根据多项证据判断失效模式，例如 SCC、疲劳、腐蚀减薄、冲刷腐蚀、磨损、过载、热疲劳、振动疲劳、润滑失效、密封失效等。
4. 排除型：说明为什么某些原因不符合证据，例如材料成分不合格、热处理异常、单纯过载、制造缺陷、单一腐蚀因素等。
5. 故障链条型：说明多个现象如何形成“原因—过程—结果”的故障演化链条。
6. 风险判断型：根据事实证据判断继续运行风险、扩展风险、泄漏风险、断裂风险或复发风险。
7. 证据支撑的建议型：只有当输入中存在明确失效原因或风险证据时，才可生成建议型 QA；建议必须直接对应证据，不能写泛泛的“加强管理、定期检查”。

禁止生成“纯方法型 QA”。

以下问题属于纯方法型，禁止生成：

* “PT、RT、UT、MT 分别有什么作用？”
* “金相检验的基本流程是什么？”
* “SEM/EDS 如何用于失效分析？”
* “失效分析一般包括哪些步骤？”
* “如何保证多样品分析的代表性？”

只有当检测方法和具体检测结果同时出现时，才可以生成“检测证据解释型 QA”。

允许的问题形式：

* “UT 发现某管段存在局部减薄，而外观检查发现内壁冲刷痕迹时，这些证据说明什么？”
* “SEM 观察到疲劳条带，且裂纹源位于应力集中区域时，为什么支持疲劳开裂判断？”
* “EDS 检出氯元素富集，同时裂纹呈分叉状穿晶扩展时，对应力腐蚀开裂判断有什么帮助？”

核心事实约束：

1. 必须严格基于输入内容生成问答，不得编造未出现的设备、材料、检测结果、工况条件、失效机理或工程结论。
2. 如果输入中没有失效结论，不得伪造最终失效机理。
3. 如果输入中只有检测现象，优先生成“证据解释型”或“风险判断型”QA。
4. 如果输入中明确给出结论，可以生成“机理判断型”QA，但 thinking 必须展示支持该结论的事实证据。
5. 如果输入中出现“成分符合标准、组织正常、硬度正常、未见明显制造缺陷”等信息，优先生成排除型 QA。
6. 如果输入中出现“裂纹从内壁启裂、外壁启裂、穿晶、沿晶、分叉、树枝状、点蚀、腐蚀产物、疲劳条带、韧窝、解理、磨损沟槽、冲刷痕迹”等信息，优先生成机理判断或证据解释 QA。
7. 如果输入中出现运行参数异常，例如振动、温度、压力、流量、泄漏、噪声、润滑油污染等，应优先生成故障链条型 QA。
8. 如果证据不足以形成故障链条，不要强行补充缺失环节。

thinking 字段要求：

1. thinking 字段写“可公开展示的证据链”，不是模型内心推理。
2. thinking 必须使用编号步骤。
3. 每一步都必须包含“事实证据”或“由事实证据推出的工程含义”。
4. thinking 通常写 4-8 条；证据较少时可以少于 4 条，但不得硬凑。
5. thinking 不要写“我认为”“文本提到”“根据原文”“从片段可知”等表达。
6. thinking 不要复述整段原文，只提炼关键证据链。
7. thinking 必须体现：

   * 已知事实；
   * 事实的工程含义；
   * 多项事实之间的关联；
   * 对其他原因的排除；
   * 可支持的结论或建议。
8. 如果不能排除其他原因，不要写排除结论。
9. 如果不能判断最终机理，应明确写“现有证据只能支持风险判断或进一步检测建议”。

answer 字段要求：

1. answer 必须精简明确，通常 1-4 句话。
2. answer 优先给出：

   * 失效机理判断；
   * 证据解释；
   * 排除结论；
   * 风险判断；
   * 故障链条判断；
   * 有证据支撑的工程建议。
3. answer 不得泛泛而谈。
4. answer 不得加入 thinking 中没有证据支撑的新信息。

输出要求：

1. 问题和答案均使用中文。
2. 专业缩写可以保留，例如 SEM、EDS、PT、RT、UT、MT、ET、SCC，但首次出现时可用中文解释。
3. 如果涉及公式、应力、硬度、腐蚀速率、寿命估算、振动频率、流速、压力、温度或风险评价等内容，公式必须使用 LaTeX 表达。
4. 不要生成重复问题。
5. 不要生成过泛问题，例如“这段文字说明了什么？”。
6. 输出必须是严格 JSON 数组，不要 Markdown，不要代码块，不要额外说明。
7. JSON 中不得出现注释、尾随逗号或非法转义字符。
8. 请严格按照最高上限 {max_new_tokens} token 限制输出，不要超过。

输出格式：

[
{{
"question": "面向通用工程设备失效分析场景的中文问题",
"thinking": [
"1. 关键事实证据。",
"2. 该证据对应的工程含义。",
"3. 第二项关键事实证据。",
"4. 多项证据之间的关联关系。",
"5. 可排除或不能排除的其他原因。",
"6. 最终可支持的机理判断、风险判断或工程建议。"
],
"answer": "简洁明确的中文答案。",
"qa_type": "事实依据型/证据解释型/机理判断型/排除型/故障链条型/风险判断型/证据支撑的建议型"
}}
]
"""

USER_PROMPT_TEMPLATE = r"""请从下面 PDF 文本片段中抽取高质量中文 QA 对。

{qa_count_instruction}

生成目标：

请优先生成“事实依据驱动的推理 QA”，而不是方法论 QA。

每个 QA 都必须围绕片段中已经出现的事实证据展开，体现：
“事实依据 → 证据解释 → 证据关联 → 排除或风险判断 → 结论/建议”。

禁止生成以下类型的问题：

1. 失效分析通用流程类问题；
2. 检测方法泛泛介绍类问题；
3. 多样品处理方法类问题；
4. 没有具体检测结果支撑的建议类问题；
5. 与输入事实无关的空泛工程管理问题；
6. 文章特指型阅读理解问题。

优先生成以下类型的 QA：

1. 事实依据型：
   围绕片段中明确出现的设备、材料、工况、检测结果、损伤形貌、运行异常生成问题。

2. 证据解释型：
   解释某个检测结果或损伤形貌说明什么。
   例如：点蚀坑、腐蚀产物、树枝状裂纹、疲劳条带、局部减薄、磨损沟槽、硬度异常、组织异常等。

3. 机理判断型：
   根据多项证据判断失效模式。
   例如：应力腐蚀开裂、疲劳、腐蚀减薄、冲刷腐蚀、磨损、过载、热疲劳、振动疲劳、润滑失效、密封失效、焊接缺陷等。

4. 排除型：
   当片段中出现“成分符合标准、组织正常、硬度正常、未见明显制造缺陷”等信息时，优先生成为什么可以排除材料不合格、热处理异常、制造缺陷等原因的 QA。

5. 故障链条型：
   当片段中同时出现多个异常现象时，生成“这些现象如何构成故障演化链条”的 QA。
   例如：润滑油污染 → 轴承磨损 → 振动升高 → 温度异常。

6. 风险判断型：
   根据已出现的事实证据判断泄漏风险、裂纹扩展风险、断裂风险、腐蚀加剧风险或复发风险。

7. 证据支撑的建议型：
   只有当片段中存在明确失效原因、风险证据或整改依据时，才可以生成建议型 QA。
   建议必须直接对应证据，不能写泛泛的“加强管理、定期检查”。

特别限制：

* 不要生成“PT/RT/UT/MT/金相/硬度/SEM/EDS 有什么作用”这类纯方法问题。
* 只有当片段中同时出现“检测方法 + 具体检测结果”时，才可以生成检测证据解释型 QA。
* 不要问“应该如何开展分析”“如何处理多个样品”“如何保证代表性”等方法论问题。
* 不要生成“该文档/该论文/该实验/该样品/上述案例/文中”等特指型问题。
* 如果片段中只有检测现象，没有最终结论，不要强行写最终失效机理。
* 如果片段中明确给出结论，可以生成机理判断型 QA，但 thinking 必须展示支持该结论的事实证据。
* answer 通常控制在 1-4 句话。
* thinking 使用 JSON 数组，每个元素是一条编号证据链步骤。
* thinking 通常写 4-8 条；证据较少时可以少于 4 条，但不得硬凑。
* thinking 不要复述整段原文，只提炼关键证据、证据含义、关联关系、排除过程和结论依据。
* answer 不得加入 thinking 中没有出现的新事实。

推荐问题形式：

* “已知……，应如何判断……？”
* “……这一检测结果说明什么？”
* “……为什么支持……失效机理判断？”
* “……为什么可以排除……原因？”
* “……之间如何形成故障链条？”
* “……对后续检修或风险控制有什么启示？”

禁止问题示例：

* “失效分析一般包括哪些步骤？”
* “SEM/EDS 在失效分析中有什么作用？”
* “金相分析如何用于设备失效分析？”
* “多件相似失效样品分析中应如何保证效率与代表性？”
* “该文档是否包含关于磨损颗粒分析的方法？”
* “本文使用了哪些实验方法？”

推荐问题示例：

* “已知材料成分合格、组织正常、内壁存在点蚀坑、裂纹从内壁启裂且呈树枝状扩展，应如何判断失效机理？”
* “裂纹起源于内壁点蚀坑附近，并呈分叉状扩展时，这些事实为什么支持应力腐蚀开裂判断？”
* “硬度和金相组织正常，但断口存在疲劳条带且运行中存在周期载荷时，为什么可以优先考虑疲劳扩展？”
* “局部壁厚减薄、内壁存在冲刷痕迹且介质流速较高时，如何判断冲刷腐蚀风险？”
* “轴承温度升高、振动增大且润滑油中存在磨粒污染时，这些证据如何构成润滑失效导致磨损加剧的故障链条？”

输出格式必须为严格 JSON 数组：

[
{{
"question": "中文问题",
"thinking": [
"1. 关键事实证据。",
"2. 该证据说明的工程含义。",
"3. 另一项关键事实证据。",
"4. 多项证据之间的关联。",
"5. 可排除或不能排除的原因。",
"6. 可支持的结论、风险判断或建议。"
],
"answer": "简短中文答案",
"qa_type": "事实依据型/证据解释型/机理判断型/排除型/故障链条型/风险判断型/证据支撑的建议型"
}}
]

PDF 文件：{source_pdf}
页码范围：{pages}

文本片段：
{text}
"""

REVIEW_PROMPT = r"""你是一名工业设备失效分析数据集质量审核专家。请检查下面 QA 数据是否适合用于微调。

你的审核目标是：只保留“事实依据驱动的工程推理 QA”，删除“方法论 QA”“空泛建议 QA”“阅读理解 QA”和“无依据推理 QA”。

审核标准：

1. 是否严格基于原始片段中的事实证据。
2. 是否存在原始片段未出现的设备、材料、工况、检测结果、失效机理或工程结论。
3. question 是否围绕具体事实证据展开，而不是围绕通用方法、流程、管理原则展开。
4. thinking 是否是公开可展示的证据链，而不是模型自我思考。
5. thinking 中每一步是否都有事实依据或清晰的工程含义。
6. answer 是否简洁、准确、工程化。
7. 是否存在无依据的最终失效机理判断。
8. 是否存在重复、空泛、低价值问题。
9. 是否已经从具体论文、实验、样品或设备编号中抽象为通用失效分析知识。
10. question 是否避免了“本文、该文档、该论文、该研究、该实验、该样品、该压力容器、该泵、该阀门、上述案例、文中”等文章特指表达。
11. JSON 格式是否正确。

必须删除以下 QA：

1. 纯方法型 QA：

   * “失效分析应如何开展？”
   * “某检测方法有什么作用？”
   * “多件样品应如何处理？”
   * “如何保证分析效率与代表性？”

2. 空泛建议型 QA：

   * 没有具体失效原因或风险证据，却提出“加强管理、定期检查、优化维护”等泛泛建议。

3. 无依据机理判断型 QA：

   * 原始片段没有足够证据，却强行判断为疲劳、SCC、腐蚀、过载、制造缺陷等。

4. 文章特指型 QA：

   * 问题依赖“本文、该文档、该样品、上述案例”等指代。

5. 阅读理解型 QA：

   * 只询问原文写了什么，而没有抽象成通用工程判断能力。

6. 检测方法泛泛介绍型 QA：

   * 只问 PT、RT、UT、MT、金相、硬度、SEM、EDS 的一般作用，而没有结合具体检测结果。

允许保留以下 QA：

1. 根据具体事实证据解释检测结果含义的 QA。
2. 根据多项证据判断失效机理的 QA。
3. 根据正常检测结果排除某类原因的 QA。
4. 根据多个异常现象构建故障链条的 QA。
5. 根据明确风险证据提出针对性建议的 QA。
6. 在证据不足时，只做证据解释、风险提示或进一步检测建议的 QA。

请只输出通过审核后的 JSON 数组。
不要解释原因。
不要输出 Markdown。
不要输出代码块。

原始片段：
{text}

待审核 QA：
{qa_json}
"""



@dataclass
class PageText:
    page: int
    text: str


@dataclass
class Chunk:
    source_pdf: str
    chunk_id: int
    page_start: int
    page_end: int
    text: str


def load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    with env_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_with_pdfplumber(pdf_path: Path, max_pages: Optional[int]) -> List[PageText]:
    if pdfplumber is None:
        return []
    pages: List[PageText] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        selected = pdf.pages[:max_pages] if max_pages else pdf.pages
        for idx, page in enumerate(selected, start=1):
            text = normalize_text(page.extract_text() or "")
            if text:
                pages.append(PageText(idx, text))
    return pages


def extract_with_pypdf(pdf_path: Path, max_pages: Optional[int]) -> List[PageText]:
    if PdfReader is None:
        return []
    pages: List[PageText] = []
    reader = PdfReader(str(pdf_path))
    selected = reader.pages[:max_pages] if max_pages else reader.pages
    for idx, page in enumerate(selected, start=1):
        text = normalize_text(page.extract_text() or "")
        if text:
            pages.append(PageText(idx, text))
    return pages


def extract_pdf_pages(pdf_path: Path, max_pages: Optional[int]) -> List[PageText]:
    pages = extract_with_pdfplumber(pdf_path, max_pages)
    if pages:
        return pages
    pages = extract_with_pypdf(pdf_path, max_pages)
    if pages:
        return pages
    raise RuntimeError("无法提取 PDF 文本：请安装 pdfplumber 或 pypdf，或确认 PDF 不是纯扫描件。")


def split_text_by_chars(text: str, chunk_chars: int) -> List[str]:
    pieces: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        piece = text[start:end]
        last_stop = max(piece.rfind("."), piece.rfind("。"), piece.rfind(";"), piece.rfind("；"))
        if last_stop > chunk_chars * 0.55:
            end = start + last_stop + 1
            piece = text[start:end]
        pieces.append(normalize_text(piece))
        start = end
    return [piece for piece in pieces if piece]


def make_chunks(pdf_path: Path, pages: List[PageText], pages_per_chunk: int, chunk_chars: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    chunk_id = 1

    for start in range(0, len(pages), pages_per_chunk):
        page_group = pages[start : start + pages_per_chunk]
        text = normalize_text(" ".join(page.text for page in page_group))
        if text:
            pieces = split_text_by_chars(text, chunk_chars) if len(text) > chunk_chars else [text]
            for piece in pieces:
                if piece:
                    chunks.append(
                        Chunk(
                            source_pdf=pdf_path.name,
                            chunk_id=chunk_id,
                            page_start=page_group[0].page,
                            page_end=page_group[-1].page,
                            text=piece,
                        )
                    )
                    chunk_id += 1
    return chunks


def build_system_prompt(max_tokens: int) -> str:
    return SYSTEM_PROMPT.replace("{max_new_tokens}", str(max_tokens))


def build_qa_count_instruction(max_pairs: Optional[int]) -> str:
    if max_pairs and max_pairs > 0:
        return f"请根据文本的信息密度自主判断应生成多少个 QA 对，但最多不要超过 {max_pairs} 个。"
    return "请根据信息密度、证据完整性和工程价值自主判断应生成多少个 QA 对；信息不足时可以少生成或不生成。"


def build_user_prompt(chunk: Chunk, max_pairs: Optional[int]) -> str:
    pages = f"{chunk.page_start}-{chunk.page_end}" if chunk.page_start != chunk.page_end else str(chunk.page_start)
    return USER_PROMPT_TEMPLATE.format(
        qa_count_instruction=build_qa_count_instruction(max_pairs),
        source_pdf=chunk.source_pdf,
        pages=pages,
        text=chunk.text,
    )


def build_review_prompt(chunk: Chunk, qa_items: List[Dict[str, Any]]) -> str:
    return REVIEW_PROMPT.format(
        text=chunk.text,
        qa_json=json.dumps(qa_items, ensure_ascii=False, indent=2),
    )


def extract_response_content(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if content is not None:
                return str(content)
        return json.dumps(response, ensure_ascii=False)

    choices = getattr(response, "choices", None) or []
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if content is not None:
            return str(content)
    return str(response)


def extract_json_array(raw: str) -> List[Any]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        arrays: List[List[Any]] = []
        pos = 0
        while True:
            start = text.find("[", pos)
            if start < 0:
                break
            try:
                value, end = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                pos = start + 1
                continue
            if isinstance(value, list):
                arrays.append(value)
            pos = start + end
        if not arrays:
            raise
        data = [item for array in arrays for item in array]

    if not isinstance(data, list):
        raise ValueError("模型输出不是 JSON 数组")
    return data


ARTICLE_SPECIFIC_PATTERNS = [
    "本文",
    "该文",
    "该论文",
    "该文档",
    "该研究",
    "该实验",
    "该样品",
    "该案例",
    "上述案例",
    "文中",
    "文章",
    "本研究",
    "本实验",
    "本试验",
]


def is_article_specific_question(question: str) -> bool:
    return any(pattern in question for pattern in ARTICLE_SPECIFIC_PATTERNS)


def normalize_generated_items(items: Iterable[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        row = dict(item)
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer:
            continue
        if is_article_specific_question(question):
            continue
        row["question"] = question
        row["answer"] = answer
        thinking = row.get("thinking")
        if isinstance(thinking, str):
            row["thinking"] = [line.strip() for line in thinking.splitlines() if line.strip()]
        rows.append(row)
    return rows


class ApiClient:
    def __init__(self, model: str, base_url: str, api_key: str, temperature: float, max_tokens: int):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = build_system_prompt(max_tokens)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return extract_response_content(response)

    def generate(self, chunk: Chunk, max_pairs: Optional[int]) -> str:
        return self.chat(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": build_user_prompt(chunk, max_pairs)},
            ]
        )

    def review(self, chunk: Chunk, qa_items: List[Dict[str, Any]]) -> str:
        return self.chat(
            [
                {"role": "system", "content": "你是严格的 JSON 数据质量审核器，只输出合法 JSON 数组。"},
                {"role": "user", "content": build_review_prompt(chunk, qa_items)},
            ]
        )


def iter_pdf_paths(input_path: Path, limit: Optional[int]) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"输入文件不是 PDF：{input_path}")
        return [input_path]

    pdfs = sorted(input_path.glob("*.pdf"))
    if limit:
        pdfs = pdfs[:limit]
    return pdfs


def load_summary_rows(summary_path: Path) -> List[Dict[str, Any]]:
    if not summary_path.exists():
        return []
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def load_resume_key(output_path: Path) -> Optional[tuple[str, int]]:
    if not output_path.exists():
        return None
    resume_key: Optional[tuple[str, int]] = None
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            source_pdf = row.get("source_pdf")
            chunk_id = row.get("chunk_id")
            if source_pdf and isinstance(chunk_id, int):
                resume_key = (source_pdf, chunk_id)
    return resume_key


def upsert_summary_row(rows: List[Dict[str, Any]], pdf_summary: Dict[str, Any]) -> None:
    source_pdf = pdf_summary.get("source_pdf")
    for index, row in enumerate(rows):
        if row.get("source_pdf") == source_pdf:
            rows[index] = pdf_summary
            return
    rows.append(pdf_summary)


def write_rows(output_path: Path, rows: List[Dict[str, Any]], append: bool) -> None:
    mode = "a" if append else "w"
    with output_path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(summary_path: Path, rows: List[Dict[str, Any]]) -> None:
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def env_value(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def make_client(args: argparse.Namespace) -> ApiClient:
    base_url = args.base_url or env_value("API_BASE_URL", "GLM_BASE_URL", "ZHIPUAI_BASE_URL")
    api_key = args.api_key or env_value("API_KEY", "GLM_API_KEY", "ZHIPUAI_API_KEY")
    model = args.model or env_value("API_MODEL", "GLM_MODEL", "ZHIPUAI_MODEL") or DEFAULT_MODEL

    if not base_url:
        raise RuntimeError("缺少 API base_url：请设置 .env 中的 API_BASE_URL/GLM_BASE_URL，或传入 --base-url。")
    if not api_key:
        raise RuntimeError("缺少 API key：请设置 .env 中的 API_KEY/GLM_API_KEY，或传入 --api-key。")
    return ApiClient(model=model, base_url=base_url, api_key=api_key, temperature=args.temperature, max_tokens=args.max_tokens)


def process_chunk(
    client: ApiClient,
    chunk: Chunk,
    max_pairs_per_chunk: Optional[int],
    review_enabled: bool,
) -> List[Dict[str, Any]]:
    start_time = time.time()
    raw = client.generate(chunk, max_pairs_per_chunk)
    print(f"    开始解析模型输出: chars={len(raw)}", flush=True)
    parsed_items = extract_json_array(raw)
    items = normalize_generated_items(parsed_items)
    print(
        f"    解析完成: parsed_items={len(parsed_items)}, kept_items={len(items)}, elapsed_total={time.time() - start_time:.1f}s",
        flush=True,
    )

    if review_enabled and items:
        review_start = time.time()
        review_raw = client.review(chunk, items)
        print(f"    开始解析审核输出: chars={len(review_raw)}", flush=True)
        items = normalize_generated_items(extract_json_array(review_raw))
        print(f"    审核完成: qa_items={len(items)}, elapsed={time.time() - review_start:.1f}s", flush=True)

    return [
        {
            **item,
            "source_pdf": chunk.source_pdf,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "chunk_id": chunk.chunk_id,
        }
        for item in items
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="通过 OpenAI 兼容 API 从 PDF 生成中文失效分析 QA 数据。")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="PDF 文件或 PDF 文件夹，默认 papers_failure_analysis")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出 JSONL 文件，默认 qa_api.jsonl")
    parser.add_argument("--model", default=None, help="模型名，默认读取 API_MODEL/GLM_MODEL，否则 glm-latest")
    parser.add_argument("--base-url", default=None, help="OpenAI 兼容 API base_url，默认读取 API_BASE_URL/GLM_BASE_URL")
    parser.add_argument("--api-key", default=None, help="OpenAI 兼容 API key，默认读取 API_KEY/GLM_API_KEY")
    parser.add_argument("--limit", type=int, default=None, help="最多处理多少篇 PDF")
    parser.add_argument("--max-pages", type=int, default=None, help="每篇 PDF 最多读取多少页")
    parser.add_argument("--pages-per-chunk", type=int, default=30, help="每个 chunk 最多包含多少页，默认 30")
    parser.add_argument("--chunk-chars", type=int, default=12000, help="单个页组过长时的字符切分上限，默认 12000")
    parser.add_argument("--pairs-per-chunk", type=int, default=None, help="每个文本块最多生成多少个 QA")
    parser.add_argument("--temperature", type=float, default=0.1, help="生成温度，默认 0.1")
    parser.add_argument("--max-tokens", type=int, default=4096, help="模型最大输出 token 数，默认 4096")
    parser.add_argument("--review", action="store_true", help="生成后再调用 REVIEW_PROMPT 审核一次")
    parser.add_argument("--resume", action="store_true", help="跳过输出文件中已有的 source_pdf + chunk_id")
    parser.add_argument("--overwrite", action="store_true", help="覆盖输出文件；未设置时默认追加")
    parser.add_argument("--sleep", type=float, default=0.0, help="每个 chunk 调用后的暂停秒数")
    return parser.parse_args()


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    error_path = output_path.with_suffix(output_path.suffix + ".errors.jsonl")
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")

    pdf_paths = iter_pdf_paths(input_path, args.limit)
    if not pdf_paths:
        raise SystemExit(f"没有找到 PDF：{input_path}")

    client = make_client(args)
    resume_key = load_resume_key(output_path) if args.resume else None
    resume_pending = resume_key is not None
    first_write = args.overwrite or not output_path.exists()
    total_rows = 0
    summary_rows = load_summary_rows(summary_path) if args.resume else []
    if args.resume and resume_key:
        print(f"Resume from after {resume_key[0]} chunk {resume_key[1]}", flush=True)

    for pdf_index, pdf_path in enumerate(pdf_paths, start=1):
        if resume_pending and resume_key and pdf_path.name != resume_key[0]:
            print(f"[{pdf_index}/{len(pdf_paths)}] Skip before resume marker PDF: {pdf_path.name}", flush=True)
            continue
        print(f"[{pdf_index}/{len(pdf_paths)}] 读取 PDF: {pdf_path.name}", flush=True)
        pdf_summary: Dict[str, Any] = {
            "source_pdf": pdf_path.name,
            "pages_with_text": 0,
            "chunks": 0,
            "success_chunks": 0,
            "failed_chunks": 0,
            "qa_pairs": 0,
        }
        if args.resume:
            for row in summary_rows:
                if row.get("source_pdf") == pdf_path.name:
                    for field in ("success_chunks", "failed_chunks", "qa_pairs"):
                        value = row.get(field)
                        if isinstance(value, int):
                            pdf_summary[field] = value
                    break

        try:
            pages = extract_pdf_pages(pdf_path, args.max_pages)
            chunks = make_chunks(pdf_path, pages, args.pages_per_chunk, args.chunk_chars)
            pdf_summary["pages_with_text"] = len(pages)
            pdf_summary["chunks"] = len(chunks)
        except Exception as exc:
            error = {"source_pdf": pdf_path.name, "stage": "extract_pdf", "error": str(exc)}
            write_rows(error_path, [error], append=True)
            pdf_summary["failed_chunks"] = 1
            pdf_summary["error"] = str(exc)
            upsert_summary_row(summary_rows, pdf_summary)
            write_summary(summary_path, summary_rows)
            print(f"  跳过：{exc}", flush=True)
            continue

        print(f"  页数文本块: {len(pages)} pages, {len(chunks)} chunks", flush=True)
        for chunk in chunks:
            key = (chunk.source_pdf, chunk.chunk_id)
            if resume_pending:
                if key == resume_key:
                    resume_pending = False
                    print(f"  Resume marker reached, skip completed chunk {chunk.chunk_id}", flush=True)
                else:
                    print(f"  Skip before resume marker chunk {chunk.chunk_id}", flush=True)
                continue
            try:
                print(
                    f"  chunk {chunk.chunk_id}/{len(chunks)}: pages={chunk.page_start}-{chunk.page_end}, chars={len(chunk.text)}",
                    flush=True,
                )
                rows = process_chunk(client, chunk, args.pairs_per_chunk, args.review)
                write_rows(output_path, rows, append=not first_write)
                first_write = False
                total_rows += len(rows)
                pdf_summary["success_chunks"] += 1
                pdf_summary["qa_pairs"] += len(rows)
                print(f"  chunk {chunk.chunk_id}: 写入 {len(rows)} 条 QA", flush=True)
            except Exception as exc:
                pdf_summary["failed_chunks"] += 1
                error = {
                    "source_pdf": chunk.source_pdf,
                    "chunk_id": chunk.chunk_id,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "stage": "generate_or_parse",
                    "error": str(exc),
                }
                write_rows(error_path, [error], append=True)
                print(f"  chunk {chunk.chunk_id}: 失败，已记录到 {error_path.name}: {exc}", flush=True)

            if args.sleep:
                time.sleep(args.sleep)

        upsert_summary_row(summary_rows, pdf_summary)
        write_summary(summary_path, summary_rows)
        print(
            f"  PDF汇总: qa_pairs={pdf_summary['qa_pairs']}, "
            f"success_chunks={pdf_summary['success_chunks']}, failed_chunks={pdf_summary['failed_chunks']}",
            flush=True,
        )

    if resume_pending and resume_key:
        print(f"Resume marker not found: {resume_key[0]} chunk {resume_key[1]}", flush=True)

    print(f"完成：本次写入 {total_rows} 条 QA -> {output_path}", flush=True)
    print(f"按 PDF 汇总：{summary_path}", flush=True)
    if error_path.exists():
        print(f"如有失败块，可查看：{error_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("用户中断。", file=sys.stderr)
        raise
