from docx import Document
import pdfplumber
import argparse
import re
import os
import sys
import io
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv


# =========================
# 1. 读取 .env 配置
# =========================
load_dotenv()

GLM_API_KEY = os.getenv("API_KEY")
GLM_BASE_URL = os.getenv("API_BASE_URL")
GLM_MODEL = os.getenv("API_MODEL") or "glm-latest"

client = OpenAI(
    api_key=GLM_API_KEY,
    base_url=GLM_BASE_URL
)

MIN_CHUNK_CHARS = 6000
MAX_CHUNK_CHARS = 12000
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# 日志配置：同时输出到文件（UTF-8，不受控制台编码影响）和控制台。
# 控制台 handler 用一个能容错编码的包装，避免在 GBK 控制台（部分 Windows Server）
# 打印中文时抛 UnicodeEncodeError（表现为刷屏 "Logging error ... in emit"）。
_log_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# 文件 handler：始终 UTF-8，保证完整日志落盘可回溯
_file_handler = logging.FileHandler(
    os.path.join(DATA_DIR, "run_word.log"), mode="a", encoding="utf-8"
)
_file_handler.setFormatter(_log_formatter)

# 控制台 handler：强制以 UTF-8 输出。优先 reconfigure，失败则用 TextIOWrapper
# 直接包装底层 buffer，彻底绕开 Python 判定的 GBK 编码（部分 Windows Server 上
# reconfigure 不足以生效，会刷屏 "Logging error ... in emit"）。
_console_stream = sys.stdout
try:
    if hasattr(_console_stream, "reconfigure"):
        _console_stream.reconfigure(encoding="utf-8", errors="replace")
    elif hasattr(_console_stream, "buffer"):
        _console_stream = io.TextIOWrapper(
            _console_stream.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
except Exception:
    # 兜底：即便包装失败，也让控制台 handler 用容错编码，不影响文件日志
    if hasattr(sys.stdout, "buffer"):
        try:
            _console_stream = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
        except Exception:
            _console_stream = sys.stdout
_console_handler = logging.StreamHandler(_console_stream)
_console_handler.setFormatter(_log_formatter)

logging.basicConfig(level=logging.INFO, handlers=[_file_handler, _console_handler])
logger = logging.getLogger("create_qa_word")

GENERATION_PROMPT =''' 
【角色设定】
你是失效分析知识工程师，专长于核电厂设备失效报告的结构化提取与知识库入库预处理。

【任务目标】
将输入的失效分析技术报告内容，提取为Markdown格式。

【强制提取字段】（必须完整填充，无信息则标注"未提及"）
1. **报告标识**：原始报告完整名称（剔除日期信息）
2. **失效部件**：
   - 部件名称、材料牌号/等级（如35CrMo、8.8级、Inconel X-750）
   - 规格参数（尺寸、硬度、强度等级）
   - 在设备中的具体位置/功能
3. **场景信息**：
   - 电厂名称（如红沿河核电厂）
   - 系统代码及全称（如CEX-凝结水抽取系统）
   - 设备位号（如H5SIR221PO）
   - 发生时间（精确到年月，含服役时长）
4. **失效机理**（200字内）：
   - 失效性质（氢脆/疲劳/脆性/腐蚀/过载等）
   - 关键检验数据（ 动态提取，禁止限定固定章节：宏观检查、成分分析、金相检验、力学性能、微观特征、能谱分析、腐蚀产物成分等）
   
5. **失效结论**： 整段保留，不要修改任何字
6. **建议措施**： 整段保留，不要修改任何字
7. **经验标签**：
   - 失效模式标签（如氢脆断裂、旋转弯曲疲劳、点蚀）
   - 材料类别（合金钢/碳钢/铸铁/镍基合金/不锈钢）
   - 系统分类
   - 设备类型（泵/阀门/齿轮箱/弹簧等）

【格式规范】
- 输出为Markdown代码块，使用以下结构：
```markdown
# [报告主标题]

## 基本信息
- **报告名称**: 
- **失效部件**: [名称]，[材料]，[规格]，[位置]
- **所属电厂**: 
- **所属系统**: [代码]-[全称]
- **发生时间**: [发现时间]，服役[时长]

## 失效分析结果

### 关键检验数据
- **宏观检查**: 
- **成分分析**: 
- **金相组织**: 
- **力学性能**: [硬度/强度具体数值及与标准对比]
- **微观特征**: 
- **数据标准化要求**：
     * 所有数值必须标注**单位**（如**15.7%**、**255HV10**、**28J**）
     * 必须包含**标准符合性判定**（符合/超标/不合格/未提及）
     * 裂纹/断口必须描述**启裂位置**和**扩展方向**


### 失效机理
[机理描述，突出关键数据]

## 失效结论
[结论]

## 后续建议
- [建议]

## 经验标签
`失效模式:XXX` `材料:XXX` `系统:XXX` `设备:XXX`

【动态提取规则】
1. **检测项目识别**：通读全文，识别所有检测章节（如"晶间腐蚀性能检验"、"氢含量检测"等），禁止遗漏
2. **数据优先级**：
   - 优先提取**与失效直接相关**的检测数据（如断口分析、腐蚀产物成分）
   - 次要提取**材料基础性能**（如常规拉伸、硬度）
3. **缺失处理**：若报告中某检测项目未开展，标注"未检测"而非省略
4. **单位统一**：确保所有数值带单位，百分比保留1位小数，硬度标注标尺（HRC/HV/HBW）

【长度控制规则】

除了失效结论和建议措施，单个字段描述不超过300字，避免向量化截断
数值类信息优先使用加粗突出（如硬度**520HV**）
删除原文中的示意图描述、寒暄语句、标准全文引用，仅保留结论性数据

【质量检查】

输出前自检：是否包含以下关键数值？
[ ] 材料牌号/等级
[ ] 具体硬度/强度数值（如**520HV**、**1700MPa**）
[ ] 服役时间（如8个月、25个月）
[ ] 失效性质判定关键词（氢脆/疲劳/脆性/腐蚀）
[ ] 标准符合性判定（符合/超标/不合格）
'''