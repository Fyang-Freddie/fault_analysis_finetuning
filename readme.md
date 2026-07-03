# QA 数据生成脚本使用说明


## 使用pdf文件进行处理

本项目使用 `create_qa.py` 从 `papers_failure_analysis` 中的 PDF 提取中文失效分析 QA 数据。

目前只支持两种模型后端：

- `qwen`：本地 `models/Qwen3-8B`，默认按文本块处理。
- `glm`：通过 API 调用 `glm-5.2`，默认将整篇 PDF 文本作为一个输入块，利用长上下文避免切块造成信息断裂。

### 1. 配置 GLM API

如果使用 `glm` 后端，请在项目根目录创建 `.env` 文件：

```env
GLM_BASE_URL=https://你的接口地址/v1
GLM_API_KEY=你的API_KEY
GLM_MODEL=glm-5.2
```

脚本会自动读取 `.env`。命令行中的 `--base-url`、`--api-key`、`--model` 会覆盖 `.env` 中的配置。

### 2. 使用本地 Qwen3-8B

断点续写，可以跳过之前已经生成过的内容：

```powershell
python create_qa.py --backend qwen --quantization 4bit --dtype bfloat16 --gpu-memory 10GiB --max-tokens 1024 --review --resume
```

全部重写：

```powershell
python create_qa.py --backend qwen --quantization 4bit --dtype bfloat16 --gpu-memory 10GiB --max-tokens 1024 --review --overwrite
```

小样本测试：

```powershell
python create_qa.py --backend qwen --limit 2 --max-pages 10 --max-tokens 2048 --quantization 4bit --dtype bfloat16 --gpu-memory 10GiB --review --overwrite --output qa_professional_sample.jsonl
```

### 3. 使用 GLM 5.2 API

断点续写：

```powershell
python create_qa.py --backend glm --max-tokens 4096 --review --resume
```

全部重写：

```powershell
python create_qa.py --backend glm --max-tokens 4096 --review --overwrite
```

小样本测试：

```powershell
python create_qa.py --backend glm --limit 2 --max-pages 5 --max-tokens 4096 --review --overwrite --output qa_glm_sample.jsonl
```

### 4. 使用 PDF 通用 API 脚本：create_qa_api.py

`create_qa_api.py` 是按 OpenAI 兼容接口新写的 API 版本，默认读取 `papers_failure_analysis` 下的 PDF，并输出到 `qa_api.jsonl`。

推荐在 `.env` 中配置：

```env
API_BASE_URL=http://120.26.36.89:18080/v1
API_KEY=你的API_KEY
API_MODEL=glm-latest
```

也兼容读取 `GLM_BASE_URL`、`GLM_API_KEY`、`GLM_MODEL`。

测试接口是否可用：

```powershell
python test.py
```

小样本生成：

```powershell
python create_qa_api.py --limit 2 --max-pages 5 --overwrite --output qa_api_sample.jsonl
```

指定接口和模型：

```powershell
python create_qa_api.py --base-url http://120.26.36.89:18080/v1 --model glm-latest --limit 2 --overwrite
```

断点续写：

```powershell
python create_qa_api.py --resume
```

全部重写：

```powershell
python create_qa_api.py --overwrite
```

如果大 PDF 出现连接错误，可以调小分块：

```powershell
python create_qa_api.py --pages-per-chunk 15 --resume
```

如需生成后再让模型审核一次：

```powershell
python create_qa_api.py --review --limit 2 --overwrite
```

### create_qa_api.py 常用参数

- `--input`：PDF 文件或 PDF 文件夹，默认 `papers_failure_analysis`。
- `--output`：输出 JSONL 文件，默认 `qa_api.jsonl`。
- `--resume`：跳过输出文件中已有的 `source_pdf + chunk_id`。
- `--overwrite`：覆盖输出文件。
- `--review`：生成后再调用模型审核一次，只保留审核通过的 QA。
- `--pages-per-chunk`：每个 chunk 包含的最大页数，默认 30。
- `--chunk-chars`：单个页组过长时的字符切分上限，默认 12000。
- `--pairs-per-chunk`：可选上限；不传时由模型根据文本信息密度自主判断生成多少个 QA。
- `--max-tokens`：模型最大输出 token 数，默认 4096。
- `--base-url`、`--api-key`、`--model`：覆盖 `.env` 中的 API 配置。


## 使用word进行处理

### 5. 使用 Word API 脚本：create_qa_word.py

`create_qa_word.py` 用于处理 Word 文档，默认读取 `papers_failure_analysis2` 下的 `.docx` 文件，并输出到 `qa_word.jsonl`。

该脚本会把整篇 Word 文档清洗后一次传给模型生成 QA。写入 JSONL 时，会为每条数据补充统一字段：

- `qa_type`：由 `--qa_type` 传入，默认空字符串。
- `source_pdf`：当前处理的 Word 文件名。
- `page_start`：固定为 `0`。
- `page_end`：固定为 `0`。
- `chunk_id`：固定为 `1`。

断点续写：

```powershell
python create_qa_word.py --write_mode resume --qa_type failure_analysis
```

`resume` 是默认模式，也可以简写为：

```powershell
python create_qa_word.py --qa_type failure_analysis
```

全部重写：

```powershell
python create_qa_word.py --write_mode rewrite --qa_type failure_analysis
```

指定输入和输出：

```powershell
python create_qa_word.py --input_folder papers_failure_analysis2 --output_file qa_word.jsonl --error_file error_log.txt --qa_type failure_analysis
```

### create_qa_word.py 常用参数

- `--input_folder`：Word 文件夹，默认 `papers_failure_analysis2`。
- `--output_file`：输出 JSONL 文件，默认 `qa_word.jsonl`。
- `--error_file`：错误日志文件，默认 `error_log.txt`。
- `--qa_type`：写入每条 JSON 数据的 `qa_type` 字段。
- `--write_mode`：写入模式，支持 `resume` 和 `rewrite`，默认 `resume`。

## 6. 文本输入模式

默认使用 `--chunk-mode auto`：

- `--backend qwen` 时等同于 `--chunk-mode chunk`，按 `--chunk-chars` 分块。
- `--backend glm` 时等同于 `--chunk-mode file`，整篇 PDF 文本一次传给模型。

也可以手动指定：

```powershell
python create_qa.py --backend glm --chunk-mode chunk
python create_qa.py --backend qwen --chunk-mode file
```

`create_qa_api.py` 默认按页分块，`--pages-per-chunk` 默认为 30。也就是超过 30 页的 PDF 会按每 30 页一个 chunk 处理；单个页组文本仍然过长时，再按 `--chunk-chars` 做保护性切分。

`create_qa_word.py` 默认不按页切分，整篇 Word 文档作为一个输入块处理，因此写出的 `chunk_id` 固定为 `1`，`page_start` 和 `page_end` 固定为 `0`。
