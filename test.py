
from openai import OpenAI

client = OpenAI(
    api_key="sk-9e9b574a25292d2de853c7e3e19b397a0af6f0b690355c8c01964e12b8f8dc4d",
    base_url="http://120.26.36.89:18080/v1"
)

models = client.models.list()

for model in models.data:
    print(model.id)

resp = client.chat.completions.create(
    model="glm-latest",  # 如果不行，换成你的模型列表里的真实模型名
    messages=[
        {"role": "user", "content": "失效分析是什么？请用中文简洁解释。"}
    ],
    temperature=0
)

print(resp.choices[0].message.content)