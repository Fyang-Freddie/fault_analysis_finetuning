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
DEFAULT_OUTPUT = "qa_qwen3_8b.jsonl"
DEFAULT_MODEL = str(Path(__file__).resolve().parent / "models" / "Qwen3-8B")
DEFAULT_GLM_MODEL = "glm-5.2"


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


SYSTEM_PROMPT = r"""你是一名核电、火电、石化行业通用设备失效分析专家，熟悉压力容器、管道、阀门、泵、风机、换热器、汽轮机辅机、压缩机、轴承、密封件、焊接结构、承压部件和机械装备的失效分析。

你的任务是从论文、技术报告、工程失效分析报告或多份相关文件中，抽取高质量中文问答数据，用于训练“工业设备失效分析问答模型”。

数据目标：

你要生成的是“可迁移的通用设备失效分析知识”和“抽象案例题”，而不是针对某一篇文章、某一个文档、某一次具体实验或某一个特定编号设备的阅读理解题。

问题不一定局限于金属材料失效，也可以覆盖腐蚀、疲劳、磨损、振动、密封失效、润滑异常、热应力、流体冲刷、制造缺陷、焊接缺陷、安装偏差、运行工况异常、维护不当、老化退化、控制或监测异常等通用设备失效问题。

必须把原文中的具体对象抽象为通用工程场景。例如：

* 不要问：“该文档是否包含关于磨损颗粒分析的具体方法？”
* 不要问：“该压力容器破裂的主要失效机理是什么？”
* 应该问：“当断口呈现疲劳条带并伴随循环载荷工况时，应如何判断裂纹扩展机制？”
* 应该问：“已知材料成分合格、组织正常、内壁存在点蚀坑、裂纹从内壁启裂且呈树枝状扩展，应如何判断失效机理？”
* 应该问：“当泵组运行中出现振动升高、轴承温度异常和润滑油污染时，应如何分析可能的故障链条？”
* 应该问：“当换热器管束局部减薄并伴随介质流速较高和腐蚀产物沉积时，应如何判断冲刷腐蚀风险？”

全局阅读与强推理要求：

1. 如果输入包含多份文件、全文内容或多个片段，必须先综合检查所有可见输入内容，再生成问答对。
2. 不要只根据单个句子机械生成 QA，应优先生成需要“多条证据综合判断”的强推理问答。
3. 可以跨段落、跨章节或跨文件整合证据，但所有证据都必须来自输入内容，不得编造。
4. 优先生成能够体现“故障现象 → 检测证据 → 工况分析 → 失效机理判断 → 排除其他原因 → 工程建议”的问答。
5. 如果不同文件或片段之间存在相似案例，应进行抽象归纳，生成更通用的工程失效分析问题，而不是重复生成多个近似问题。
6. 如果证据不足以支持最终失效机理判断，只能生成“证据解释型”“现象描述型”或“进一步检测建议型”QA，不得强行下结论。

核心要求：

1. 必须严格基于输入内容生成问答，不得编造未出现的设备、材料、检测结果、工况条件、失效机理或工程结论。

2. 如果证据不足，只能生成“证据解释型”“现象描述型”或“进一步检测建议型”QA，不要强行判断最终失效原因。

3. 问题必须贴近核电、火电、石化行业的通用设备失效分析场景，优先覆盖：

   * 失效机理判断：应力腐蚀开裂、疲劳、蠕变、腐蚀减薄、冲刷腐蚀、磨损、过载、热疲劳、振动疲劳、密封失效、润滑失效、焊接缺陷、制造缺陷、安装偏差等；
   * 裂纹与断裂分析：启裂位置、扩展方向、穿晶/沿晶、分叉、树枝状、疲劳条带、韧窝、解理、脆性断裂、塑性变形等；
   * 设备运行异常分析：振动升高、温度异常、泄漏、压力波动、流量异常、噪声、效率下降、堵塞、结垢、磨损、卡涩、密封失效等；
   * 检测证据解释：PT、RT、UT、MT、ET、金相、硬度、化学成分、SEM、EDS、腐蚀产物、断口形貌、振动频谱、油液分析、壁厚测量、泄漏检测、运行参数趋势等；
   * 排除性分析：为什么不是材料成分不合格、为什么不是热处理异常、为什么不是单纯过载、为什么不是单一腐蚀因素、为什么需要考虑运行工况或维护因素等；
   * 工程建议：检修、更换、在线监测、材料替换、应力控制、介质控制、防腐、润滑管理、密封改进、焊接改进、工况优化、定期检测、风险评估等。

4. thinking 字段写“可公开展示的证据链”，通常用 5-10 条编号步骤表达；证据较少时不得硬凑步数，证据充分时应展开关键中间推理、排除过程和工程含义。

5. thinking 字段不得写模型自我思考，不得写“我认为”“文本提到”“根据原文”等表达，不得写无依据猜测。

6. thinking 必须体现“证据 → 推理 → 结论”的工程逻辑。可以包含排除过程，但必须有输入内容支持。

7. answer 字段必须精简明确，优先给出失效机理、原因判断、证据解释、排除结论、风险判断或工程建议。

8. 问题和答案均使用中文。专业缩写可以保留，例如 SEM、EDS、PT、RT、UT、MT、ET、SCC，但首次出现时可用中文解释。

9. 如果涉及公式、应力、硬度、腐蚀速率、寿命估算、振动频率、流速、压力、温度或风险评价等内容，公式必须使用 LaTeX 表达。

10. 不要生成重复问题，不要生成过泛的问题，例如“这段文字说明了什么？”。

11. 不要生成文章特指型问题，问题中不得出现“本文、该文、该论文、该文档、该研究、该实验、该样品、该压力容器、该泵、该阀门、上述案例、文中”等依赖原文指代的表达。

12. 可以保留材料牌号、设备类型、检测方法、失效形貌、工况条件等具有泛化价值的技术信息；但不要保留无泛化价值的文献标题、作者、DOI、图号、表号、具体页码、具体设备编号或事故日期。

13. 每个 QA 应尽量对应一个明确的工程判断能力，例如机理识别、证据解释、原因排除、风险判断、故障链条分析或整改建议。

14. 输出必须是严格 JSON 数组，不要 Markdown，不要代码块，不要额外说明。

15. JSON 中不得出现注释、尾随逗号或非法转义字符。

16. 请你严格按照最高上限 {max_new_tokens} token 限制输出，不要超过。


输出格式：

[
{
"question": "面向通用工程设备失效分析场景的中文问题",
"thinking": [
"1. 关键故障现象、检测结果或运行异常。",
"2. 由该证据可以推导出的工程含义。",
"3. 第二项关键证据及其工程含义。",
"4. 多项证据之间的关联关系。",
"5. 对不符合证据的其他原因进行排除。",
"6. 最终可支持的机理判断、风险判断或工程建议。"
],
"answer": "简洁明确的中文答案。"
}
]
"""


def build_system_prompt(max_tokens: int) -> str:
    return SYSTEM_PROMPT.replace("{max_new_tokens}", str(max_tokens))


USER_PROMPT_TEMPLATE = """请从下面 PDF 文本片段中抽取高质量中文 QA 对。
{qa_count_instruction}
每个 QA 对必须使用以下 JSON 结构：

[
{{
"question": "中文问题",
"thinking": "1. 证据步骤\n2. 证据步骤\n3. 证据关联\n4. 排除或风险判断\n5. 推理结论",
"answer": "简短中文答案",
"qa_type": "事实型/证据解释型/机理判断型/排除型/方法型/建议型"
}}
]

优先生成以下类型的 QA：

1. 事实型：设备、材料、工况、检测方法、检测结果；
2. 证据解释型：某个检测结果说明什么；
3. 机理判断型：根据多项证据判断失效模式；
4. 排除型：为什么不是材料不合格、制造缺陷、疲劳、过载、热处理异常等；
5. 方法型：PT、RT、金相、硬度、SEM、EDS等检测方法分别证明什么；
6. 建议型：根据失效原因提出预防、检修或改进建议。

生成规则：

* 如果片段中只有检测现象，没有最终结论，不要强行写最终失效机理。
* 如果片段中明确给出结论，可以生成机理判断型 QA。
* 如果片段中有“成分符合标准、组织正常、硬度正常”等信息，优先生成排除型 QA。
* 如果片段中有“裂纹从内壁启裂、外壁启裂、穿晶、沿晶、分叉、树枝状、点蚀、腐蚀产物”等信息，优先生成裂纹机理分析 QA。
* 如果片段中有检测方法，必须至少生成一个“检测方法意义”相关 QA。
* answer 不要太长，通常控制在 1-4 句话。
* thinking 通常写 5-10 条编号步骤，证据较少时可以少于 5 条但不得硬凑；证据充分时应展开关键中间推理、证据关联、排除过程和工程含义。
* thinking 不要复述整段原文，只提炼关键证据链。
* 必须把具体论文内容抽象成通用工程知识，不要生成“该文档/该论文/该实验/该压力容器/该泵/该阀门”的特指型阅读理解问题。
* question 应该像工程师面对相似证据时会提出的问题，尽量写成“已知……应如何判断……？”、“……说明什么？”、“……为何会促进……？”、“……之间有什么关联？”。
* 禁止问题示例：
  - “该文档是否包含关于磨损颗粒分析的具体方法？”
  - “该压力容器破裂的主要失效机理是什么？”
  - “本文使用了哪些实验方法？”
* 推荐问题示例：
  - “已知材料成分合格、组织正常、内壁存在点腐蚀坑、裂纹从内壁启裂且呈树枝状扩展，应如何判断失效机理？”
  - “内表层形变组织为何会促进应力腐蚀裂纹萌生？”
  - “腐蚀浅斑与应力腐蚀裂纹之间存在什么关联？”
  - “SEM/EDS 同时发现腐蚀产物和脆性断口形貌时，对失效机理判断有什么帮助？”

Few-shot 示例：

[
{{
"question": "已知材料成分符合标准、基体组织正常、内壁存在点蚀坑，且裂纹从内壁启裂并呈树枝状穿晶扩展，应如何判断失效机理？",
"thinking": "1. 成分符合标准，材料错用或成分不合格可能性较低\n2. 基体组织正常，整体热处理或冶金异常证据不足\n3. 点蚀坑位于内壁，说明介质环境参与了裂纹萌生\n4. 裂纹从内壁启裂，启裂位置与介质接触面一致\n5. 树枝状分叉和穿晶扩展符合应力腐蚀开裂特征\n6. 材料、应力和腐蚀环境三要素具备，应优先判断为 SCC",
"answer": "应优先判断为应力腐蚀开裂（SCC）或腐蚀-应力耦合作用导致的开裂。",
"qa_type": "机理判断型"
}},
{{
"question": "内表层形变组织为何会促进应力腐蚀裂纹萌生？",
"thinking": "1. 表层形变组织通常伴随冷作硬化和位错密度升高\n2. 局部硬度升高会提高残余拉应力或应力集中敏感性\n3. 形变区的钝化膜稳定性可能降低，更易发生局部腐蚀\n4. 在腐蚀介质存在时，形变层更容易成为裂纹萌生区\n5. 应力集中和局部腐蚀共同提高 SCC 敏感性",
"answer": "内表层形变组织会提高局部应力集中和腐蚀敏感性，使腐蚀介质更容易在表层缺陷处诱发应力腐蚀裂纹。",
"qa_type": "证据解释型"
}},
{{
"question": "腐蚀浅斑、点蚀坑与应力腐蚀裂纹之间通常存在什么关联？",
"thinking": "1. 腐蚀浅斑说明材料表面已经受到介质作用\n2. 点蚀坑会造成局部几何缺口和应力集中\n3. 点蚀坑底部的化学环境更容易恶化\n4. 在拉应力存在时，点蚀坑可转化为裂纹萌生源\n5. 若裂纹从腐蚀坑附近启裂并发生分叉扩展，应考虑 SCC",
"answer": "腐蚀浅斑和点蚀坑通常是介质参与失效的证据，点蚀坑可作为应力集中和局部腐蚀源，进一步诱发应力腐蚀裂纹萌生。",
"qa_type": "证据解释型"
}}
]

PDF 文件：{source_pdf}
页码范围：{pages}

文本片段：
{text}
"""

REVIEW_PROMPT = """你是一名失效分析数据集质量审核专家。请检查下面 QA 数据是否适合用于微调。

审核标准：

1. 是否严格基于原始片段；
2. 是否存在无依据失效机理判断；
3. thinking 是否是公开可展示的证据链，而不是模型自我思考；
4. answer 是否简洁、准确、工程化；
5. 是否存在重复、空泛或低价值问题；
6. 是否已经从具体论文/实验/设备中抽象为通用失效分析知识；
7. question 是否避免了“本文、该文档、该论文、该研究、该实验、该样品、该压力容器、该泵、该阀门、上述案例、文中”等文章特指表达；
8. JSON 格式是否正确。

请只输出通过审核后的 JSON 数组。删除低质量 QA、文章特指型 QA、阅读理解型 QA，不要解释原因。

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


def make_chunks(pdf_path: Path, pages: List[PageText], chunk_chars: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    buffer: List[str] = []
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    chunk_id = 1

    def flush() -> None:
        nonlocal buffer, page_start, page_end, chunk_id
        text = normalize_text(" ".join(buffer))
        if text:
            chunks.append(
                Chunk(
                    source_pdf=pdf_path.name,
                    chunk_id=chunk_id,
                    page_start=page_start or 0,
                    page_end=page_end or page_start or 0,
                    text=text,
                )
            )
            chunk_id += 1
        buffer = []
        page_start = None
        page_end = None

    for page in pages:
        text = page.text
        if len(text) > chunk_chars:
            flush()
            start = 0
            while start < len(text):
                end = min(start + chunk_chars, len(text))
                piece = text[start:end]
                last_stop = max(piece.rfind("."), piece.rfind("。"), piece.rfind(";"), piece.rfind("；"))
                if last_stop > chunk_chars * 0.55:
                    end = start + last_stop + 1
                    piece = text[start:end]
                chunks.append(
                    Chunk(
                        source_pdf=pdf_path.name,
                        chunk_id=chunk_id,
                        page_start=page.page,
                        page_end=page.page,
                        text=normalize_text(piece),
                    )
                )
                chunk_id += 1
                start = end
            continue

        prospective_len = len(" ".join(buffer)) + len(text)
        if buffer and prospective_len > chunk_chars:
            flush()
        if page_start is None:
            page_start = page.page
        page_end = page.page
        buffer.append(text)

    flush()
    return chunks


def make_file_chunk(pdf_path: Path, pages: List[PageText]) -> List[Chunk]:
    text = normalize_text(" ".join(page.text for page in pages))
    if not text:
        return []
    return [
        Chunk(
            source_pdf=pdf_path.name,
            chunk_id=1,
            page_start=pages[0].page,
            page_end=pages[-1].page,
            text=text,
        )
    ]


def resolve_chunk_mode(args: argparse.Namespace) -> str:
    if args.chunk_mode != "auto":
        return args.chunk_mode
    return "file" if args.backend == "glm" else "chunk"


def extract_json_array(raw: str) -> List[Dict[str, Any]]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end < start:
            raise
        data = json.loads(text[start : end + 1])

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
]


def is_article_specific_question(question: str) -> bool:
    return any(pattern in question for pattern in ARTICLE_SPECIFIC_PATTERNS)


OPTIONAL_QA_FIELDS = [
    "qa_type",
    "domain",
    "sub_domain",
    "equipment",
    "failure_mode",
    "analysis_method",
    "source_pdf",
]


def normalize_analysis_steps(item: Dict[str, Any]) -> List[str]:
    raw_steps = item.get("analysis_steps")
    if raw_steps is None:
        raw_steps = item.get("thinking")
    if isinstance(raw_steps, list):
        return [str(step).strip() for step in raw_steps if str(step).strip()]
    if isinstance(raw_steps, str):
        return [step.strip() for step in raw_steps.splitlines() if step.strip()]
    return []


def validate_qa_items(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        analysis_steps = normalize_analysis_steps(item)
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            continue
        if is_article_specific_question(question):
            continue
        key = (question, answer)
        if key in seen:
            continue
        seen.add(key)
        cleaned_item = {
            "question": question,
            "analysis_steps": analysis_steps,
            "answer": answer,
        }
        for field in OPTIONAL_QA_FIELDS:
            value = item.get(field)
            if value in (None, "", []):
                continue
            cleaned_item[field] = value
        cleaned.append(cleaned_item)
    return cleaned


def build_qa_count_instruction(max_pairs: Optional[int]) -> str:
    if max_pairs and max_pairs > 0:
        return f"请根据文本的信息密度自主判断应生成多少个 QA 对，但最多不要超过 {max_pairs} 个。"
    return (
        "请根据文本的信息密度、证据完整性和工程价值自主判断应生成多少个 QA 对；"
        "信息不足时可以少生成或不生成，不要为了凑数量生成空泛、重复或低价值 QA。"
    )


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


class QwenClient:
    def generate(self, chunk: Chunk, max_pairs: Optional[int]) -> str:
        raise NotImplementedError

    def review(self, chunk: Chunk, qa_items: List[Dict[str, str]]) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(QwenClient):
    def __init__(self, model: str, base_url: Optional[str], api_key: Optional[str], temperature: float, max_tokens: int):
        from openai import OpenAI

        kwargs: Dict[str, Any] = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self.client = OpenAI(**kwargs)
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
        return response.choices[0].message.content or ""

    def generate(self, chunk: Chunk, max_pairs: Optional[int]) -> str:
        return self.chat(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": build_user_prompt(chunk, max_pairs)},
            ]
        )

    def review(self, chunk: Chunk, qa_items: List[Dict[str, str]]) -> str:
        return self.chat(
            [
                {"role": "system", "content": "你是严格的 JSON 数据质量审核器，只输出合法 JSON 数组。"},
                {"role": "user", "content": build_review_prompt(chunk, qa_items)},
            ]
        )


class TransformersClient(QwenClient):
    def __init__(
        self,
        model: str,
        temperature: float,
        max_new_tokens: int,
        dtype: str,
        gpu_memory: Optional[str],
        cpu_memory: Optional[str],
        quantization: str,
    ):
        model_ref = normalize_model_ref(model)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "当前 Python 环境缺少 transformers/torch，无法直接从 Hugging Face 加载 Qwen3-8B。\n"
                "请先安装依赖，例如：\n"
                "  pip install -U transformers accelerate pdfplumber pypdf\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
                "如果没有 NVIDIA GPU，可改装 CPU 版 torch，但 Qwen3-8B 会非常慢。\n"
                "也可以配置 .env 后使用 --backend glm 调用 GLM 5.2 API。"
            ) from exc

        torch_dtype = resolve_torch_dtype(torch, dtype)
        max_memory = build_max_memory(torch, gpu_memory, cpu_memory)
        quantization_config = build_quantization_config(quantization, torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
        load_kwargs = {
            "dtype": torch_dtype,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if max_memory:
            load_kwargs["max_memory"] = max_memory
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config

        self.model = AutoModelForCausalLM.from_pretrained(model_ref, **load_kwargs)
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.system_prompt = build_system_prompt(max_new_tokens)
        print(f"模型加载完成: {model_ref}", flush=True)
        device_map = getattr(self.model, "hf_device_map", self.model.device)
        print(f"模型设备映射: {device_map}", flush=True)
        warn_if_cpu_offload(device_map)

    def generate_messages(self, messages: List[Dict[str, str]], stage: str) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        input_tokens = int(inputs.input_ids.shape[-1])
        print(
            f"    Qwen {stage}开始: input_tokens={input_tokens}, max_new_tokens={self.max_new_tokens}",
            flush=True,
        )
        start_time = time.time()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
        )
        elapsed = time.time() - start_time
        generated = outputs[0][inputs.input_ids.shape[-1] :]
        output_tokens = int(generated.shape[-1])
        speed = output_tokens / elapsed if elapsed > 0 else 0
        print(
            f"    Qwen {stage}完成: output_tokens={output_tokens}, elapsed={elapsed:.1f}s, speed={speed:.2f} tok/s",
            flush=True,
        )
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate(self, chunk: Chunk, max_pairs: Optional[int]) -> str:
        return self.generate_messages(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": build_user_prompt(chunk, max_pairs)},
            ],
            stage="生成",
        )

    def review(self, chunk: Chunk, qa_items: List[Dict[str, str]]) -> str:
        return self.generate_messages(
            [
                {"role": "system", "content": "你是严格的 JSON 数据质量审核器，只输出合法 JSON 数组。"},
                {"role": "user", "content": build_review_prompt(chunk, qa_items)},
            ],
            stage="审核",
        )


def looks_like_windows_path(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", value)) or "\\" in value


def resolve_torch_dtype(torch_module: Any, dtype: str) -> Any:
    dtype = dtype.lower()
    if dtype == "auto":
        return "auto"
    if dtype in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if dtype in {"fp16", "float16", "half"}:
        return torch_module.float16
    if dtype in {"fp32", "float32"}:
        return torch_module.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def build_max_memory(torch_module: Any, gpu_memory: Optional[str], cpu_memory: Optional[str]) -> Optional[Dict[Any, str]]:
    if not gpu_memory and not cpu_memory:
        return None

    max_memory: Dict[Any, str] = {}
    if torch_module.cuda.is_available() and gpu_memory:
        for device_idx in range(torch_module.cuda.device_count()):
            max_memory[device_idx] = gpu_memory
    elif gpu_memory:
        print("警告：指定了 --gpu-memory，但 torch.cuda.is_available() 为 False，GPU 显存限制不会生效。", flush=True)

    if cpu_memory:
        max_memory["cpu"] = cpu_memory
    return max_memory or None


def build_quantization_config(quantization: str, compute_dtype: Any) -> Optional[Any]:
    quantization = quantization.lower()
    if quantization in {"none", "no", "false"}:
        return None
    if quantization not in {"4bit", "8bit"}:
        raise ValueError(f"Unsupported quantization: {quantization}")

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "使用量化需要新版 transformers 和 bitsandbytes。\n"
            "请先执行：python -m pip install -U transformers accelerate bitsandbytes"
        ) from exc

    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def warn_if_cpu_offload(device_map: Any) -> None:
    if not isinstance(device_map, dict):
        return
    cpu_parts = [name for name, device in device_map.items() if str(device).lower() == "cpu"]
    cuda_parts = [name for name, device in device_map.items() if str(device).lower() in {"0", "cuda", "cuda:0"}]
    if cpu_parts and cuda_parts:
        print(
            f"提示：检测到 {len(cpu_parts)} 个模块在 CPU、{len(cuda_parts)} 个模块在 GPU。"
            " 生成时会频繁 CPU/GPU 传输，可能比纯 GPU 慢很多；12GB 显卡建议使用 --quantization 4bit。",
            flush=True,
        )


def normalize_model_ref(model: str) -> str:
    candidate_path = Path(model).expanduser()
    is_existing_local_path = candidate_path.exists()
    is_explicit_local_path = looks_like_windows_path(model) or model.startswith(".") or model.startswith("/")

    if is_existing_local_path or is_explicit_local_path:
        model_path = candidate_path
        if not model_path.exists():
            raise FileNotFoundError(
                f"本地模型目录不存在：{model_path}\n"
                "请确认路径是否写对，或改用 Hugging Face repo 名称：--model Qwen/Qwen3-8B。\n"
                "如果模型还没下载，可先执行：\n"
                "  huggingface-cli download Qwen/Qwen3-8B --local-dir D:\\freddie\\finetuning\\models\\Qwen3-8B"
            )
        if not model_path.is_dir():
            raise NotADirectoryError(f"模型路径不是目录：{model_path}")
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"模型目录缺少 config.json：{model_path}\n"
                "请确认该目录是完整的 Hugging Face 模型目录，而不是上一级目录或未下载完成的目录。"
            )
        return str(model_path.resolve())
    return model


def iter_pdf_paths(input_path: Path, limit: Optional[int]) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"输入文件不是 PDF：{input_path}")
        return [input_path]

    pdfs = sorted(input_path.glob("*.pdf"))
    return pdfs[:limit] if limit else pdfs


def load_done_keys(output_path: Path) -> set:
    done = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            source = row.get("source_pdf")
            chunk_id = row.get("chunk_id")
            if source and chunk_id:
                done.add((source, int(chunk_id)))
    return done


def make_client(args: argparse.Namespace) -> QwenClient:
    if args.backend == "qwen":
        return TransformersClient(
            model=args.model,
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
            dtype=args.dtype,
            gpu_memory=args.gpu_memory,
            cpu_memory=args.cpu_memory,
            quantization=args.quantization,
        )

    base_url = (
        args.base_url
        or os.getenv("GLM_BASE_URL")
        or os.getenv("ZHIPUAI_BASE_URL")
    )
    if not base_url:
        raise RuntimeError("使用 --backend glm 时必须在 .env 中配置 GLM_BASE_URL，或通过命令行传入 --base-url。")
    api_key = args.api_key or os.getenv("GLM_API_KEY") or os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise RuntimeError("使用 --backend glm 时必须在 .env 中配置 GLM_API_KEY，或通过命令行传入 --api-key。")
    return OpenAICompatibleClient(
        model=args.model,
        base_url=base_url,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


def write_rows(
    output_path: Path,
    rows: List[Dict[str, Any]],
    append: bool,
) -> None:
    mode = "a" if append else "w"
    with output_path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(summary_path: Path, rows: List[Dict[str, Any]]) -> None:
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def summarize_existing_output(output_path: Path) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    if not output_path.exists():
        raise FileNotFoundError(f"输出文件不存在：{output_path}")

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            source_pdf = row.get("source_pdf") or "UNKNOWN"
            counts[source_pdf] = counts.get(source_pdf, 0) + 1

    return [
        {"source_pdf": source_pdf, "qa_pairs": count}
        for source_pdf, count in sorted(counts.items(), key=lambda item: item[0])
    ]


def process_chunk(
    client: QwenClient,
    chunk: Chunk,
    max_pairs_per_chunk: Optional[int],
    output_format: str,
    review_enabled: bool,
) -> List[Dict[str, Any]]:
    start_time = time.time()
    raw = client.generate(chunk, max_pairs_per_chunk)
    print(f"    开始解析模型输出: chars={len(raw)}", flush=True)
    items = validate_qa_items(extract_json_array(raw))
    print(f"    解析完成: qa_items={len(items)}, elapsed_total={time.time() - start_time:.1f}s", flush=True)

    if review_enabled and items:
        review_start = time.time()
        review_raw = client.review(chunk, items)
        print(f"    开始解析审核输出: chars={len(review_raw)}", flush=True)
        reviewed_items = validate_qa_items(extract_json_array(review_raw))
        print(
            f"    审核完成: before={len(items)}, after={len(reviewed_items)}, elapsed={time.time() - review_start:.1f}s",
            flush=True,
        )
        items = reviewed_items

    rows: List[Dict[str, Any]] = []
    for item in items:
        if output_format == "simple":
            rows.append(item)
        else:
            rows.append(
                {
                    **item,
                    "source_pdf": chunk.source_pdf,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "chunk_id": chunk.chunk_id,
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use Qwen3-8B to extract Chinese failure-analysis QA pairs from PDFs."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="PDF 文件或 PDF 文件夹，默认 papers_failure_analysis")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出 JSONL，默认 qa_qwen3_8b.jsonl")
    parser.add_argument(
        "--backend",
        choices=["qwen", "glm"],
        default="qwen",
        help="模型后端：qwen 使用本地 Qwen3-8B；glm 使用 .env 中配置的 GLM 5.2 API。",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face 模型名或本地模型路径，默认 Qwen/Qwen3-8B")
    parser.add_argument("--base-url", default=None, help="GLM API 服务地址；默认读取 .env 中的 GLM_BASE_URL")
    parser.add_argument("--api-key", default=None, help="GLM API key；默认读取 .env 中的 GLM_API_KEY")
    parser.add_argument("--limit", type=int, default=None, help="最多处理多少篇 PDF")
    parser.add_argument("--max-pages", type=int, default=None, help="每篇 PDF 最多读取多少页，调试时很有用")
    parser.add_argument("--chunk-chars", type=int, default=6000, help="每个模型输入块的最大字符数")
    parser.add_argument(
        "--pairs-per-chunk",
        type=int,
        default=None,
        help="每个输入块最多生成多少个 QA；不传时由模型根据信息密度自主判断数量。",
    )
    parser.add_argument("--temperature", type=float, default=0.1, help="生成温度，建议 0-0.3")
    parser.add_argument("--max-tokens", type=int, default=2048, help="模型最大输出 token 数")
    parser.add_argument("--max-new-tokens", dest="max_tokens", type=int, help="同 --max-tokens，便于按 transformers 习惯传参")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        default="bfloat16",
        help="Hugging Face 模型加载精度，默认 bfloat16。",
    )
    parser.add_argument(
        "--gpu-memory",
        default=None,
        help="每张 GPU 可用显存上限，例如 10GiB。12GB 显卡建议 9GiB-10GiB。",
    )
    parser.add_argument(
        "--cpu-memory",
        default=None,
        help="CPU offload 内存上限，例如 48GiB。显存放不下完整模型时建议设置。",
    )
    parser.add_argument(
        "--quantization",
        choices=["none", "4bit", "8bit"],
        default="none",
        help="bitsandbytes 量化。12GB 显卡跑 Qwen3-8B 建议 4bit。",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="每次模型调用后的暂停秒数")
    parser.add_argument(
        "--output-format",
        choices=["with_source", "simple"],
        default="with_source",
        help="with_source 会附带 PDF/页码/chunk_id；simple 只输出模型生成的 QA 字段",
    )
    parser.add_argument("--resume", action="store_true", help="跳过输出文件中已有的 source_pdf + chunk_id")
    parser.add_argument("--overwrite", action="store_true", help="覆盖输出文件；未设置时默认追加")
    parser.add_argument("--stats-only", action="store_true", help="只统计已有输出 JSONL 中每篇 PDF 的 QA 数量，不调用模型")
    parser.add_argument("--review", action="store_true", help="生成 QA 后调用 REVIEW_PROMPT 进行二次审核，只保留审核通过的 QA")
    parser.add_argument(
        "--chunk-mode",
        choices=["auto", "chunk", "file"],
        default="auto",
        help="文本输入模式：chunk 按 --chunk-chars 分块；file 将整篇 PDF 作为一个输入块；auto 在 glm 后端使用 file，其它后端使用 chunk。",
    )
    args = parser.parse_args()
    if args.backend == "glm" and args.model == DEFAULT_MODEL:
        args.model = os.getenv("GLM_MODEL") or os.getenv("ZHIPUAI_MODEL") or DEFAULT_GLM_MODEL
    return args


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    args = parse_args()
    chunk_mode = resolve_chunk_mode(args)
    input_path = Path(args.input)
    output_path = Path(args.output)
    error_path = output_path.with_suffix(output_path.suffix + ".errors.jsonl")
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")

    if args.stats_only:
        summary_rows = summarize_existing_output(output_path)
        write_summary(summary_path, summary_rows)
        total = sum(row["qa_pairs"] for row in summary_rows)
        for row in summary_rows:
            print(f"{row['source_pdf']}: {row['qa_pairs']} QA")
        print(f"总计：{total} QA")
        print(f"按 PDF 汇总：{summary_path}")
        return

    pdf_paths = iter_pdf_paths(input_path, args.limit)
    if not pdf_paths:
        raise SystemExit(f"没有找到 PDF：{input_path}")

    client = make_client(args)
    done_keys = load_done_keys(output_path) if args.resume else set()
    first_write = args.overwrite or not output_path.exists()
    total_rows = 0
    summary_rows: List[Dict[str, Any]] = []

    for pdf_index, pdf_path in enumerate(pdf_paths, start=1):
        print(f"[{pdf_index}/{len(pdf_paths)}] 读取 PDF: {pdf_path.name}")
        pdf_summary: Dict[str, Any] = {
            "source_pdf": pdf_path.name,
            "pages_with_text": 0,
            "chunks": 0,
            "success_chunks": 0,
            "failed_chunks": 0,
            "qa_pairs": 0,
        }
        try:
            pages = extract_pdf_pages(pdf_path, args.max_pages)
            chunks = make_file_chunk(pdf_path, pages) if chunk_mode == "file" else make_chunks(pdf_path, pages, args.chunk_chars)
            pdf_summary["pages_with_text"] = len(pages)
            pdf_summary["chunks"] = len(chunks)
        except Exception as exc:
            error = {"source_pdf": pdf_path.name, "stage": "extract_pdf", "error": str(exc)}
            write_rows(error_path, [error], append=True)
            pdf_summary["failed_chunks"] = 1
            pdf_summary["error"] = str(exc)
            summary_rows.append(pdf_summary)
            write_summary(summary_path, summary_rows)
            print(f"  跳过：{exc}")
            continue

        print(f"  页数文本块: {len(pages)} pages, {len(chunks)} chunks")
        for chunk in chunks:
            key = (chunk.source_pdf, chunk.chunk_id)
            if key in done_keys:
                print(f"  跳过已完成 chunk {chunk.chunk_id}")
                continue

            try:
                print(
                    f"  chunk {chunk.chunk_id}/{len(chunks)}: pages={chunk.page_start}-{chunk.page_end}, chars={len(chunk.text)}",
                    flush=True,
                )
                rows = process_chunk(client, chunk, args.pairs_per_chunk, args.output_format, args.review)
                write_rows(output_path, rows, append=not first_write)
                first_write = False
                total_rows += len(rows)
                pdf_summary["success_chunks"] += 1
                pdf_summary["qa_pairs"] += len(rows)
                print(f"  chunk {chunk.chunk_id}: 写入 {len(rows)} 条 QA")
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
                print(f"  chunk {chunk.chunk_id}: 失败，已记录到 {error_path.name}: {exc}")

            if args.sleep:
                time.sleep(args.sleep)

        summary_rows.append(pdf_summary)
        write_summary(summary_path, summary_rows)
        print(
            f"  PDF汇总: qa_pairs={pdf_summary['qa_pairs']}, "
            f"success_chunks={pdf_summary['success_chunks']}, failed_chunks={pdf_summary['failed_chunks']}",
            flush=True,
        )

    print(f"完成：本次写入 {total_rows} 条 QA -> {output_path}")
    print(f"按 PDF 汇总：{summary_path}")
    if error_path.exists():
        print(f"如有失败块，可查看：{error_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("用户中断。", file=sys.stderr)
        raise
