# QA 数据生成项目脚本使用说明

本项目包含若干独立脚本，用于下载失效分析资料、从 PDF 或 Word 文档生成中文 QA 数据，以及测试 API 是否可用。

## 1. 环境配置

API 版本脚本建议在项目根目录创建 `.env` 文件：

```env
API_BASE_URL=http://你的接口地址/v1
API_KEY=你的API_KEY
API_MODEL=glm-latest
```

`create_qa.py` 的 GLM 后端使用下面这些变量：

```env
GLM_BASE_URL=https://你的接口地址/v1
GLM_API_KEY=你的API_KEY
GLM_MODEL=glm-5.2
```

`create_qa_api.py` 会优先读取 `API_BASE_URL`、`API_KEY`、`API_MODEL`，也兼容 `GLM_BASE_URL`、`GLM_API_KEY`、`GLM_MODEL`。

## 2. create_qa.py

`create_qa.py` 用于从 PDF 文件或 PDF 文件夹生成 QA 数据，支持两种后端：

- `qwen`：使用本地 `models/Qwen3-8B`。
- `glm`：通过 GLM API 调用模型。

默认输入目录是 `papers_failure_analysis`，默认输出文件是 `qa_qwen3_8b.jsonl`。

### 使用本地 Qwen

```powershell
python create_qa.py --backend qwen --input papers_failure_analysis --output qa_qwen3_8b.jsonl --quantization 4bit --dtype bfloat16 --gpu-memory 10GiB --max-tokens 1024 --review --overwrite
```

小样本测试：

```powershell
python create_qa.py --backend qwen --input papers_failure_analysis --limit 2 --max-pages 10 --output qa_sample.jsonl --quantization 4bit --dtype bfloat16 --gpu-memory 10GiB --review --overwrite
```

### 使用 GLM API

```powershell
python create_qa.py --backend glm --input papers_failure_analysis --output qa_glm.jsonl --max-tokens 4096 --review --overwrite
```

断点续写：

```powershell
python create_qa.py --backend glm --input papers_failure_analysis --output qa_glm.jsonl --max-tokens 4096 --review --resume
```

### 常用参数

- `--input`：PDF 文件或 PDF 文件夹，默认 `papers_failure_analysis`。
- `--output`：输出 JSONL 文件，默认 `qa_qwen3_8b.jsonl`。
- `--backend`：模型后端，支持 `qwen` 和 `glm`。
- `--model`：本地模型路径或模型名。
- `--base-url`、`--api-key`：覆盖 `.env` 中的 GLM API 配置。
- `--limit`：最多处理多少篇 PDF。
- `--max-pages`：每篇 PDF 最多读取多少页。
- `--chunk-mode`：文本输入模式，支持 `auto`、`chunk`、`file`。
- `--chunk-chars`：分块模式下每个输入块的最大字符数，默认 6000。
- `--pairs-per-chunk`：每个输入块最多生成多少个 QA。
- `--max-tokens`：模型最大输出 token 数，默认 2048。
- `--resume`：跳过输出文件中已有的 `source_pdf + chunk_id`。
- `--overwrite`：覆盖输出文件；未设置时默认追加。
- `--review`：生成后再调用模型审核，只保留审核通过的 QA。
- `--stats-only`：只统计已有输出 JSONL 中每篇 PDF 的 QA 数量，不调用模型。

## 3. create_qa_api.py

`create_qa_api.py` 是推荐的 PDF API 版本，用 OpenAI 兼容接口从 PDF 文件或 PDF 文件夹生成 QA 数据。

默认输入目录是 `papers_failure_analysis`，默认输出文件是 `qa_api.jsonl`。

### 基本用法

```powershell
python create_qa_api.py --input papers_failure_analysis --output qa_api_facts.jsonl --overwrite
```

处理单个 PDF：

```powershell
python create_qa_api.py --input "papers_failure_analysis\example.pdf" --output qa_api.jsonl --overwrite
```

小样本测试：

```powershell
python create_qa_api.py --input papers_failure_analysis --limit 2 --max-pages 5 --output qa_api_sample.jsonl --overwrite
```

断点续写：

```powershell
python create_qa_api.py --input papers_failure_analysis --output qa_api_facts.jsonl --resume
```

大 PDF 请求不稳定时，可以调小每个 chunk 的页数：

```powershell
python create_qa_api.py --input papers_failure_analysis --output qa_api.jsonl --pages-per-chunk 15 --resume
```

### 常用参数

- `--input`：PDF 文件或 PDF 文件夹，默认 `papers_failure_analysis`。
- `--output`：输出 JSONL 文件，默认 `qa_api.jsonl`。
- `--model`：模型名，默认读取 `API_MODEL` 或 `GLM_MODEL`，否则使用 `glm-latest`。
- `--base-url`、`--api-key`：覆盖 `.env` 中的 API 配置。
- `--limit`：最多处理多少篇 PDF。
- `--max-pages`：每篇 PDF 最多读取多少页。
- `--pages-per-chunk`：每个 chunk 最多包含多少页，默认 30。
- `--chunk-chars`：单个页组过长时的字符切分上限，默认 12000。
- `--pairs-per-chunk`：每个文本块最多生成多少个 QA。
- `--temperature`：生成温度，默认 0.1。
- `--max-tokens`：模型最大输出 token 数，默认 4096。
- `--review`：生成后再调用模型审核一次。
- `--resume`：跳过输出文件中已有的 `source_pdf + chunk_id`。
- `--overwrite`：覆盖输出文件；未设置时默认追加。
- `--sleep`：每个 chunk 调用后的暂停秒数。

## 4. create_qa_word.py

`create_qa_word.py` 用于从 Word 文档生成 QA 数据。它按文件夹读取 `.docx` 文件，默认输入目录是 `papers_failure_analysis2`，默认输出文件是 `qa_word.jsonl`。

### 基本用法

```powershell
python create_qa_word.py --input_folder papers_failure_analysis2 --output_file qa_word.jsonl --error_file error_log.txt --qa_type failure_analysis --write_mode rewrite
```

断点续写：

```powershell
python create_qa_word.py --input_folder papers_failure_analysis2 --output_file qa_word.jsonl --error_file error_log.txt --qa_type failure_analysis --write_mode resume
```

如果只想处理单个 Word 文件，可以把该 `.docx` 放到一个单独目录中，再把该目录传给 `--input_folder`。

### 常用参数

- `--input_folder`：Word 文件夹，默认 `papers_failure_analysis2`。
- `--output_file`：输出 JSONL 文件，默认 `qa_word.jsonl`。
- `--error_file`：错误日志文件，默认 `error_log.txt`。
- `--qa_type`：写入每条 JSON 数据的 `qa_type` 字段。
- `--write_mode`：写入模式，支持 `resume` 和 `rewrite`，默认 `resume`。

## 5. download_paper.py

`download_paper.py` 用于按内置关键词从 Semantic Scholar、OpenAlex 和 Unpaywall 搜索开放获取 PDF，并下载到 `papers_failure_analysis2`。

直接运行：

```powershell
python download_paper.py
```

输出内容：

- PDF 文件保存到 `papers_failure_analysis2`。
- 元数据保存到 `papers_failure_analysis2\metadata.csv`。

该脚本没有命令行参数；如需修改关键词、邮箱或保存目录，需要编辑脚本中的 `QUERIES`、`EMAIL`、`SAVE_DIR`。

## 6. download_paper2.py

`download_paper2.py` 是较简化的 OpenAlex 下载脚本，按内置关键词搜索开放获取 PDF，并下载到 `papers_failure_analysis2`。

直接运行：

```powershell
python download_paper2.py
```

输出内容：

- PDF 文件保存到 `papers_failure_analysis2`。
- 元数据保存到 `papers_failure_analysis2\metadata.csv`。

该脚本没有命令行参数；如需修改关键词、邮箱或保存目录，需要编辑脚本中的 `KEYWORDS`、`EMAIL`、`SAVE_DIR`。

## 7. test.py

`test.py` 用于测试 API 接口是否能列出模型并完成一次简单对话。

运行：

```powershell
python test.py
```

注意：`test.py` 当前直接在代码中初始化 OpenAI 客户端，不读取 `.env`。使用前请确认脚本中的 `api_key`、`base_url` 和 `model` 已改成你自己的配置。

## 8. 输出文件说明

QA 生成脚本主要输出 JSONL 文件，每行是一条 QA 数据。常见字段包括：

- `question`：中文问题。
- `thinking`：可公开展示的证据链或推理步骤。
- `answer`：中文答案。
- `qa_type`：QA 类型。
- `source_pdf`：来源文件名。
- `page_start`、`page_end`、`chunk_id`：来源页码或文本块信息。

部分脚本还会生成辅助文件：

- `*.errors.jsonl`：生成或解析失败的记录。
- `*.summary.json`：生成结果统计。
- `error_log.txt`：Word 处理错误日志。
