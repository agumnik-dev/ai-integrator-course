"""
hello_claude.py — урок 0: первый вызов Claude API.

ЗАЧЕМ ЭТОТ ФАЙЛ
    Консольная утилита: задаёт вопрос модели Claude и печатает ответ
    вместе со «счётом»: сколько токенов ушло и сколько это стоило.
    Это минимальный каркас, из которого вырастет любая AI-интеграция.

КЛАССИЧЕСКИЙ FLOW РАБОТЫ С LLM (одинаков во всех будущих проектах)

    ШАГ 1. Клиент
        client = anthropic.Anthropic(api_key=...)
        Объект, через который идут все запросы к API. Создаётся один раз.

    ШАГ 2. Запрос → response
        response = client.messages.create(model=..., max_tokens=..., messages=[...])
        Отправляем модели список сообщений, получаем ответ.
        Модель НЕ помнит прошлых вызовов: всю историю диалога шлём каждый раз сами.

    ШАГ 3. Использование response
        response.content      — СПИСОК блоков (text, thinking, позже tool_use).
                                Текст берём только из блоков type == "text"
                                (функция extract_text), никогда content[0].text.
        response.stop_reason  — почему модель остановилась:
                                "end_turn"   — закончила сама, всё хорошо;
                                "max_tokens" — ответ ОБРЕЗАН, это нужно проверять.
        response.usage        — токены на вход и выход → стоимость запроса.

    Всё остальное в файле — обвязка: проверка ключа и аргументов, ошибки API.

ЗАПУСК
    uv run python 00-setup/hello_claude.py "Что такое MCP? Два предложения."
"""

import sys
import os
import anthropic
from dotenv import load_dotenv


MODEL = "claude-haiku-5-5"  # самая быстрая и дешёвая модель — для учёбы
PRICE_INPUT_PER_MTOK = 0.10  # $ за 1M входных токенов
PRICE_OUTPUT_PER_MTOK = 0.50  # $ за 1M выходных токенов
MAX_TOKENS = 500  # потолок длины ответа


class ConfigError(Exception):
    """Ошибка настройки: нет ключа, нет аргумента и т.п."""

    pass


def get_api_key() -> str:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:  # ловит и None, и пустую строку из .env.example
        raise ConfigError(
            "Не найден ANTHROPIC_API_KEY. Скопируйте .env.example в .env и впишите ключ."
        )
    return key


def extract_text(response: anthropic.types.Message) -> str:
    """Собирает текст из всех текстовых блоков ответа модели."""
    return "".join(block.text for block in response.content if block.type == "text")


def main() -> int:
    load_dotenv()  # читает .env, кладёт ключ в переменные окружения

    # check right arguments
    if len(sys.argv) < 2:
        print('Using: uv run python 00-setup/hello_claude.py "your question"')
        sys.exit(1)

    question = sys.argv[1]

    try:
        api_key = get_api_key()
    except ConfigError as e:
        print(f"Ошибка настройки: {e}", file=sys.stderr)
        return 1

    try:
        # ===== ШАГ 1: КЛИЕНТ — точка входа во все запросы к Claude =====
        client = anthropic.Anthropic(api_key=api_key, max_retries=2, timeout=30.0)

        # ===== ШАГ 2: ЗАПРОС → RESPONSE — ради этого всё и затевается =====
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": question}],
        )
    except anthropic.AuthenticationError:
        print(
            "Ключ API не принят. Проверьте ANTHROPIC_API_KEY в .env.", file=sys.stderr
        )
        return 1
    except anthropic.NotFoundError:
        print(f"Модель {MODEL} не найдена. Проверьте название.", file=sys.stderr)
        return 1
    # ... RateLimitError, BadRequestError, InternalServerError ...
    except anthropic.APIStatusError as e:  # всё остальное с HTTP-кодом
        print(f"API вернул ошибку {e.status_code}: {e.message}", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError:  # включая таймаут
        print("Нет связи с API. Проверьте интернет.", file=sys.stderr)
        return 1

    # ===== ШАГ 3: ИСПОЛЬЗОВАНИЕ RESPONSE =====
    # 3a. usage → стоимость запроса
    cost_input = (response.usage.input_tokens / 1_000_000) * PRICE_INPUT_PER_MTOK
    cost_output = (response.usage.output_tokens / 1_000_000) * PRICE_OUTPUT_PER_MTOK
    total_cost = cost_input + cost_output

    # 3b. content → текст ответа (только блоки type == "text")
    text = extract_text(response)
    print(text or "(модель не вернула текста)")

    # 3c. stop_reason → не обрезан ли ответ (ошибки API нет, но ответ неполный!)
    if response.stop_reason == "max_tokens":
        print("\nОтвет обрезан: модель упёрлась в max_tokens.")

    print(f"Модель: {response.model}")
    print(
        f"Токены: вход {response.usage.input_tokens}, выход {response.usage.output_tokens}"
    )
    print(f"Стоимость: ${total_cost:.6f}")
    print(f"Остановка: {response.stop_reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
