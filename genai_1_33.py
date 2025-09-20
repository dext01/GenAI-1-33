import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "sambanovasystems/SambaLingo-Russian-Chat"

def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    """
    Загружает модель и токенизатор Hugging Face.
    """
    print("Загрузка модели (это может занять несколько минут при первом запуске)...")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    print("✅ Модель загружена.")
    return tok, mdl


def parse_request(request: str) -> tuple[int, str]:
    """
    Достаёт количество советов и тему из русскоязычного запроса.
    """
    if not isinstance(request, str) or not request.strip():
        raise ValueError("Запрос должен быть непустой строкой.")
    text = request.lower().strip()

    # число советов
    m = re.search(r"\d+", text)
    num = int(m.group()) if m else 5

    # тема
    topic = ""
    for kw in (" по ", " о ", " про "):
        if kw in text:
            topic = text.split(kw, 1)[1]
            break
    topic = topic.strip().strip('?.!,"«»')
    return num, topic


def generate_advice(request: str, tokenizer, model) -> str:
    """
    Генерирует советы на русском языке по запросу.
    """
    num, topic = parse_request(request)
    if topic:
        user_msg = f"Дай {num} советов по теме: {topic}. Отвечай строго по-русски."
    else:
        user_msg = f"Дай {num} полезных советов. Отвечай строго по-русски."

    prompt = f"<|user|>\n{user_msg}</s>\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=max(120, num * 80),
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # выделяем только нужное количество пунктов
    tips, lines = [], text.splitlines()
    for ln in lines:
        s = ln.strip()
        if any(s.startswith(f"{i}.") or s.startswith(f"{i})") for i in range(1, num + 1)):
            tips.append(s)
        if len(tips) == num:
            break
    return "\n".join(tips) if tips else text
