import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Final

import google.generativeai as genai
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DEFAULT_PROVIDER: Final[str] = "gemini"
DEFAULT_GEMINI_MODEL: Final[str] = "gemini-2.5-flash"
DEFAULT_XAI_MODEL: Final[str] = "grok-4-latest"
DEFAULT_TEMPERATURE: Final[float] = 0.7
HISTORY_LIMIT: Final[int] = 10
SYSTEM_PROMPT: Final[str] = (
    "Ты полезный ассистент в Telegram. "
    "Отвечай на языке пользователя, будь вежливым, кратким и полезным."
)

GEMINI_MODELS: Final[tuple[str, ...]] = (
    "gemini-2.5-flash",
)

XAI_MODELS: Final[tuple[str, ...]] = (
    "grok-4-latest",
)


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Не задана переменная окружения: {name}")
    return value


def set_env_key(path: str, key: str, value: str) -> None:
    env_path = Path(path)
    lines = env_path.read_text(encoding="utf-8-sig").splitlines() if env_path.exists() else []
    out = []
    replaced = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{key}="):
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(line)

    if not replaced:
        out.append(f"{key}={value}")

    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def parse_temperature(value: str | None, fallback: float = DEFAULT_TEMPERATURE) -> float:
    if value is None:
        return fallback
    try:
        parsed = float(value.replace(",", ".").strip())
    except ValueError:
        return fallback
    return max(0.0, min(1.0, parsed))


def append_history(context: ContextTypes.DEFAULT_TYPE, role: str, text: str) -> None:
    history = context.chat_data.setdefault("history", [])
    history.append({"role": role, "text": text})
    if len(history) > HISTORY_LIMIT:
        del history[: len(history) - HISTORY_LIMIT]


def build_prompt(context: ContextTypes.DEFAULT_TYPE, user_text: str) -> str:
    history = context.chat_data.get("history", [])
    history_lines = []
    for item in history:
        prefix = "User" if item.get("role") == "user" else "Assistant"
        history_lines.append(f"{prefix}: {item.get('text', '').strip()}")

    history_block = "\n".join(history_lines) if history_lines else "(empty)"
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Conversation history (recent messages):\n{history_block}\n\n"
        f"Current user request:\n{user_text}"
    )


def explain_provider_error(exc: Exception, provider: str, model_name: str) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    short = " ".join(message.split())[:240]
    lower = message.lower()

    if "not found" in lower or ("model" in lower and ("not" in lower or "invalid" in lower)):
        return (
            f"Ошибка модели {provider} для '{model_name}'. "
            f"Укажите корректную модель в .env и перезапустите бота. Детали: {short}"
        )

    if "api key" in lower or "permission" in lower or "unauthorized" in lower or "403" in lower:
        return f"Ошибка API-ключа/прав {provider}. Детали: {short}"

    if "quota" in lower or "429" in lower or "rate" in lower:
        return f"Превышение квоты/лимита запросов {provider}. Детали: {short}"

    return f"Ошибка запроса к {provider}: {short}"


def generate_with_gemini(model: genai.GenerativeModel, prompt: str, temperature: float) -> str:
    response = model.generate_content(
        prompt,
        generation_config={"temperature": temperature},
    )
    return (response.text or "Не удалось получить ответ от Gemini.").strip()


def generate_with_xai(api_key: str, model_name: str, prompt: str, temperature: float) -> str:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    req = urllib.request.Request(
        url="https://api.x.ai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        details = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code}: {details[:400]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e.reason}") from e

    data = json.loads(raw)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Unexpected xAI response: {raw[:300]}")

    message = choices[0].get("message") or {}
    content = message.get("content")

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        text = "".join(text_parts).strip()
    else:
        text = (content or "").strip()

    return text or "Не удалось получить ответ от xAI."


def get_available_models(provider: str) -> tuple[str, ...]:
    return XAI_MODELS if provider == "xai" else GEMINI_MODELS


def set_runtime_target(application: Application, provider: str, model_name: str) -> None:
    if provider == "xai":
        xai_api_key = os.getenv("XAI_API_KEY")
        if not xai_api_key:
            raise RuntimeError("Не задана переменная окружения: XAI_API_KEY")
        application.bot_data["xai_api_key"] = xai_api_key
    else:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise RuntimeError("Не задана переменная окружения: GEMINI_API_KEY")
        genai.configure(api_key=gemini_api_key, transport="rest")
        application.bot_data["gemini_model"] = genai.GenerativeModel(model_name)

    application.bot_data["provider"] = provider
    application.bot_data["model_name"] = model_name


def persist_model(provider: str, model_name: str) -> None:
    set_env_key(".env", "MODEL_PROVIDER", provider)
    env_key = "XAI_MODEL" if provider == "xai" else "GEMINI_MODEL"
    set_env_key(".env", env_key, model_name)


def resolve_model_selection(raw: str) -> tuple[str, str] | None:
    normalized = " ".join(raw.strip().lower().replace("-", " ").split())
    aliases = {
        "gemini": ("gemini", "gemini-2.5-flash"),
        "gemini 2.5": ("gemini", "gemini-2.5-flash"),
        "gemini 2.5 flash": ("gemini", "gemini-2.5-flash"),
        "grok": ("xai", "grok-4-latest"),
        "grok 4": ("xai", "grok-4-latest"),
        "grok 4 latest": ("xai", "grok-4-latest"),
    }
    if normalized in aliases:
        return aliases[normalized]

    if raw in GEMINI_MODELS:
        return ("gemini", raw)
    if raw in XAI_MODELS:
        return ("xai", raw)
    return None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.chat_data["history"] = []
    provider = context.application.bot_data.get("provider", DEFAULT_PROVIDER)
    model_name = context.application.bot_data.get("model_name", DEFAULT_GEMINI_MODEL)
    await update.message.reply_text(
        "Привет! Я @shmalselonmuskbot с AI-интеграцией. "
        f"Провайдер: {provider}. Модель: {model_name}."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Команды:\n"
        "/start - запуск\n"
        "/help - помощь\n"
        "/model - текущая модель\n"
        "/model <name> - сменить модель\n\n"
        "/temp - текущая температура (пределы: 0..2)\n"
        "/temp <0..2> - сменить температуру\n\n"
        "Отправьте любое текстовое сообщение, и я отвечу."
    )


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    provider = context.application.bot_data.get("provider", DEFAULT_PROVIDER)
    current = context.application.bot_data.get("model_name", DEFAULT_GEMINI_MODEL)
    available_gemini = ", ".join(GEMINI_MODELS)
    available_xai = ", ".join(XAI_MODELS)

    if not context.args:
        await update.message.reply_text(
            f"Провайдер: {provider}\n"
            f"Текущая модель: {current}\n"
            f"Модели Gemini: {available_gemini}\n"
            f"Модели xAI: {available_xai}\n"
            "Использование: /model <name>\n"
            "Примеры: /model gemini 2.5, /model grok 4"
        )
        return

    requested = " ".join(context.args).strip()
    resolved = resolve_model_selection(requested)
    if not resolved:
        await update.message.reply_text(
            f"Неизвестная модель: {requested}\n"
            f"Модели Gemini: {available_gemini}\n"
            f"Модели xAI: {available_xai}"
        )
        return

    target_provider, target_model = resolved
    try:
        set_runtime_target(context.application, target_provider, target_model)
        persist_model(target_provider, target_model)
        switched_label = "grok 4" if target_provider == "xai" else "gemini 3"
        await update.message.reply_text(
            f"Переключено на {switched_label}."
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to switch model: %s", exc)
        await update.message.reply_text(explain_provider_error(exc, target_provider, target_model))


async def temp_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    current = float(context.application.bot_data.get("temperature", DEFAULT_TEMPERATURE))

    if not context.args:
        await update.message.reply_text(
            f"Текущая температура: {current:.2f}\n"
            "Использование: /temp <число от 0 до 1>\n"
            "Пример: /temp 0.7"
        )
        return

    try:
        parsed = float(context.args[0].replace(",", "."))
    except ValueError:
        await update.message.reply_text("Некорректное значение. Укажите число от 0 до 1.")
        return

    if not (0.0 <= parsed <= 1.0):
        await update.message.reply_text("Температура должна быть в диапазоне от 0 до 1.")
        return

    context.application.bot_data["temperature"] = parsed
    set_env_key(".env", "AI_TEMPERATURE", f"{parsed:.2f}")
    await update.message.reply_text(f"Температура установлена: {parsed:.2f}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    user_text = update.message.text.strip()
    if not user_text:
        await update.message.reply_text("Пожалуйста, отправьте текстовое сообщение.")
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    provider = context.application.bot_data.get("provider", DEFAULT_PROVIDER)
    model_name = context.application.bot_data.get("model_name", DEFAULT_GEMINI_MODEL)
    temperature = float(context.application.bot_data.get("temperature", DEFAULT_TEMPERATURE))
    prompt = build_prompt(context, user_text)

    try:
        if provider == "xai":
            api_key = context.application.bot_data["xai_api_key"]
            text = generate_with_xai(api_key, model_name, prompt, temperature)
        else:
            model = context.application.bot_data["gemini_model"]
            text = generate_with_gemini(model, prompt, temperature)

        append_history(context, "user", user_text)
        append_history(context, "assistant", text)

        await update.message.reply_text(text[:4096])
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error while calling %s: %s", provider, exc)
        await update.message.reply_text(explain_provider_error(exc, provider, model_name))


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled error: %s", context.error)


def main() -> None:
    load_env_file()

    telegram_token = get_required_env("TELEGRAM_BOT_TOKEN")
    provider = os.getenv("MODEL_PROVIDER", DEFAULT_PROVIDER).strip().lower()

    application = Application.builder().token(telegram_token).build()

    if provider == "xai":
        xai_api_key = get_required_env("XAI_API_KEY")
        model_name = os.getenv("XAI_MODEL", DEFAULT_XAI_MODEL)
        application.bot_data["xai_api_key"] = xai_api_key
    else:
        gemini_api_key = get_required_env("GEMINI_API_KEY")
        model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        genai.configure(api_key=gemini_api_key, transport="rest")
        application.bot_data["gemini_model"] = genai.GenerativeModel(model_name)

    application.bot_data["provider"] = provider
    application.bot_data["model_name"] = model_name
    application.bot_data["temperature"] = parse_temperature(os.getenv("AI_TEMPERATURE"))

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("model", model_command))
    application.add_handler(CommandHandler("temp", temp_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(on_error)

    logger.info("Bot started with provider=%s model=%s", provider, model_name)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
