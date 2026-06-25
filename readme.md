# QA 数据生成脚本使用说明

本项目使用 `create_qa.py` 从 `papers_failure_analysis` 中的 PDF 提取中文失效分析 QA 数据。

目前只支持两种模型后端：

- `qwen`：本地 `models/Qwen3-8B`，默认按文本块处理。
- `glm`：通过 API 调用 `glm-5.2`，默认将整篇 PDF 文本作为一个输入块，利用长上下文避免切块造成信息断裂。

## 1. 配置 GLM API

如果使用 `glm` 后端，请在项目根目录创建 `.env` 文件：

```env
GLM_BASE_URL=https://你的接口地址/v1
GLM_API_KEY=你的API_KEY
GLM_MODEL=glm-5.2
```

脚本会自动读取 `.env`。命令行中的 `--base-url`、`--api-key`、`--model` 会覆盖 `.env` 中的配置。

## 2. 使用本地 Qwen3-8B

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

## 3. 使用 GLM 5.2 API

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

## 4. 文本输入模式

默认使用 `--chunk-mode auto`：

- `--backend qwen` 时等同于 `--chunk-mode chunk`，按 `--chunk-chars` 分块。
- `--backend glm` 时等同于 `--chunk-mode file`，整篇 PDF 文本一次传给模型。

也可以手动指定：

```powershell
python create_qa.py --backend glm --chunk-mode chunk
python create_qa.py --backend qwen --chunk-mode file
```

## 5. 常用参数

- `--input`：PDF 文件或 PDF 文件夹，默认 `papers_failure_analysis`。
- `--output`：输出 JSONL 文件，默认 `qa_qwen3_8b.jsonl`。
- `--resume`：跳过输出文件中已有的 `source_pdf + chunk_id`。
- `--overwrite`：覆盖输出文件。
- `--review`：生成后再调用模型审核一次，只保留审核通过的 QA。
- `--pairs-per-chunk`：可选上限；不传时由模型根据文本信息密度自主判断生成多少个 QA。
- `--max-tokens`：模型最大输出 token 数。
- `--stats-only`：只统计已有输出，不调用模型。
