# 🤖 GenAI-1-33 — Генератор советов на русском языке

Этот проект реализует генерацию полезных советов на русском языке с помощью модели [SambaLingo-Russian-Chat](https://huggingface.co/sambanovasystems/SambaLingo-Russian-Chat).  
Код можно запускать как в локальной среде Python, так и в **Google Colab** (настоятельно рекомендуется).

---

## 🚀 Возможности
- Генерация советов **по любой теме** (учёба, здоровье, спорт и т.д.)
- Поддержка **любого количества советов** (автоматически определяется из запроса)
- Полностью **русскоязычный ответ**
- Запуск через `.py` файл или в Google Colab

---

## 📂 Структура проекта
- `genai_1_33.py` — основной модуль с функциями:
  - `load_model_and_tokenizer()` — загрузка модели и токенизатора
  - `parse_request()` — извлечение числа и темы из запроса
  - `generate_advice()` — генерация советов
  - `run_interactive()` — интерактивный режим в консоли
- `GenAI-1-33.ipynb` — пример работы в **Google Colab**
- `.gitignore` — исключает ненужные файлы (кеши, веса моделей, окружения)

---

## ▶️ Быстрый старт

### 🔹 Запуск в Google Colab
Нажмите на кнопку, чтобы открыть проект в Colab и протестировать:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dext01/GenAI-1-33/blob/main/GenAI-1-33.ipynb)

---

### 🔹 Локальный запуск
1. Склонируйте репозиторий:
   ```bash
   git clone https://github.com/dext01/GenAI-1-33.git
   cd GenAI-1-33
