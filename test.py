import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# 读取同目录下的 .env（project/.env）
load_dotenv(Path(__file__).resolve().parent / ".env")

base_url = os.getenv("API_BASE_URL")
api_key = os.getenv("API_KEY")
model = os.getenv("API_MODEL") or "glm-latest"

if not base_url or not api_key:
    raise SystemExit("缺少 API_BASE_URL / API_KEY，请检查 project/.env 是否已配置。")

print(f"连接: base_url={base_url}, model={model}")

client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0)

print("\n[1] 列出模型:")
models = client.models.list()
for m in models.data:
    print("  -", m.id)

print(f"\n[2] 调用 chat (model={model}):")
resp = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": "失效分析是什么？请用中文简洁解释。"}
    ],
    temperature=0,
)
print(resp.choices[0].message.content)
