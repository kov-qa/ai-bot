"""
AI Telegram Bot (Gemini + xAI Grok).

Запуск: python bot.py
Конфиг: .env (см. .env.example)
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Final

import google.generativeai as genai
from PIL import Image
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DEFAULT_PROVIDER: Final[str] = "gemini"
DEFAULT_GEMINI_MODEL: Final[str] = "gemini-3-flash-preview"
DEFAULT_XAI_MODEL: Final[str] = "grok-4-latest"
DEFAULT_TEMPERATURE: Final[float] = 0.7
DEFAULT_PROVIDER_TIMEOUT: Final[int] = 90
HISTORY_LIMIT: Final[int] = 20
DEFAULT_COOLDOWN: Final[float] = 2.0
TG_CHUNK: Final[int] = 4000

DEFAULT_SYSTEM_PROMPT: Final[str] = (
    "Ты полезный ассистент в Telegram. "
    "Отвечай на языке пользователя, будь вежливым, кратким и полезным. "
    "ВАЖНО: отвечай простым текстом без какой-либо разметки. "
    "Не используй markdown, не выделяй слова звёздочками (**), подчёркиваниями (_), "
    "обратными кавычками (`), решётками (#), не делай маркированные списки символами "
    "* или -. Если нужен список — используй просто новые строки или нумерацию '1.'. "
    "Никаких эмодзи без явной просьбы пользователя."
)

GEMINI_MODELS: Final[tuple[str, ...]] = (
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
)
XAI_MODELS: Final[tuple[str, ...]] = (
    "grok-4-latest",
    "grok-3-latest",
)

ENV_PATH: Final[str] = ".env"

# Runtime registry для несериализуемых объектов (gRPC клиенты, api keys).
_RUNTIME: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# .env helpers
# ---------------------------------------------------------------------------
def load_env_file(path: str = ENV_PATH) -> None:
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
    out: list[str] = []
    replaced = False
    for line in lines:
        if line.strip().startswith(f"{key}="):
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{key}={value}")
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def parse_int_list(raw: str | None) -> set[int]:
    if not raw:
        return set()
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part.lstrip("-").isdigit():
            out.add(int(part))
    return out


def parse_temperature(value: str | None, fallback: float = DEFAULT_TEMPERATURE) -> float:
    if value is None:
        return fallback
    try:
        parsed = float(value.replace(",", ".").strip())
    except ValueError:
        return fallback
    return max(0.0, min(1.0, parsed))


# ---------------------------------------------------------------------------
# Per-chat state
# ---------------------------------------------------------------------------
def chat_history(context: ContextTypes.DEFAULT_TYPE) -> list[dict[str, Any]]:
    return context.chat_data.setdefault("history", [])


def append_history(context: ContextTypes.DEFAULT_TYPE, role: str, text: str) -> None:
    history = chat_history(context)
    history.append({"role": role, "text": text, "ts": int(time.time())})
    if len(history) > HISTORY_LIMIT:
        del history[: len(history) - HISTORY_LIMIT]


def chat_system_prompt(context: ContextTypes.DEFAULT_TYPE) -> str:
    return context.chat_data.get("system_prompt") or DEFAULT_SYSTEM_PROMPT


def chat_temperature(context: ContextTypes.DEFAULT_TYPE) -> float:
    val = context.chat_data.get("temperature")
    if val is None:
        val = context.application.bot_data.get("temperature", DEFAULT_TEMPERATURE)
    return float(val)


def bump_stats(context: ContextTypes.DEFAULT_TYPE, key: str, delta: int = 1) -> None:
    stats = context.chat_data.setdefault("stats", {})
    stats[key] = stats.get(key, 0) + delta


def remember_chat(context: ContextTypes.DEFAULT_TYPE, chat_id: int, title: str) -> None:
    chats = context.application.bot_data.setdefault("known_chats", {})
    chats[chat_id] = {"title": title, "last_seen": int(time.time())}


def is_admin(context: ContextTypes.DEFAULT_TYPE, user_id: int) -> bool:
    admins: set[int] = context.application.bot_data.get("admin_ids", set())
    return not admins or user_id in admins


def is_chat_allowed(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> bool:
    allowed: set[int] = context.application.bot_data.get("allowed_chats", set())
    return not allowed or chat_id in allowed


def cooldown_check(context: ContextTypes.DEFAULT_TYPE, user_id: int) -> float:
    """Return seconds left to wait, or 0 if allowed."""
    cooldown: float = context.application.bot_data.get("cooldown", DEFAULT_COOLDOWN)
    if cooldown <= 0:
        return 0.0
    last: dict[int, float] = context.application.bot_data.setdefault("last_call", {})
    now = time.monotonic()
    prev = last.get(user_id, 0.0)
    if now - prev < cooldown:
        return cooldown - (now - prev)
    last[user_id] = now
    return 0.0


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_prompt(context: ContextTypes.DEFAULT_TYPE, user_text: str) -> str:
    history = chat_history(context)
    lines = [
        f"{'User' if item.get('role') == 'user' else 'Assistant'}: "
        f"{(item.get('text') or '').strip()}"
        for item in history
    ]
    history_block = "\n".join(lines) if lines else "(empty)"
    return (
        f"{chat_system_prompt(context)}\n\n"
        f"Conversation history (recent messages):\n{history_block}\n\n"
        f"Current user request:\n{user_text}"
    )


# ---------------------------------------------------------------------------
# Provider calls (sync — wrap via asyncio.to_thread)
# ---------------------------------------------------------------------------
def explain_provider_error(exc: Exception, provider: str, model_name: str) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    short = " ".join(message.split())[:240]
    lower = message.lower()
    if "not found" in lower or ("model" in lower and ("not" in lower or "invalid" in lower)):
        return (
            f"Ошибка модели {provider} для '{model_name}'. "
            f"Укажи корректную модель в .env. Детали: {short}"
        )
    if "api key" in lower or "permission" in lower or "unauthorized" in lower or "403" in lower:
        return f"Ошибка API-ключа/прав {provider}. Детали: {short}"
    if "quota" in lower or "429" in lower or "rate" in lower:
        return f"Лимит/квота {provider}. Детали: {short}"
    if "timeout" in lower or "timed out" in lower:
        return f"Таймаут провайдера ({provider}, model={model_name}). Попробуй позже."
    return f"Ошибка запроса {provider}: {short}"


def generate_with_gemini(
    model: genai.GenerativeModel,
    prompt: str,
    temperature: float,
    image: Image.Image | None = None,
    audio: tuple[str, bytes] | None = None,
) -> str:
    parts: list[Any] = [prompt]
    if image is not None:
        parts.append(image)
    if audio is not None:
        mime, data = audio
        parts.append({"mime_type": mime, "data": data})
    response = model.generate_content(
        parts if len(parts) > 1 else prompt,
        generation_config={"temperature": temperature},
        request_options={"timeout": DEFAULT_PROVIDER_TIMEOUT},
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
        with urllib.request.urlopen(req, timeout=DEFAULT_PROVIDER_TIMEOUT) as resp:
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
        text_parts = [
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        ]
        text = "".join(text_parts).strip()
    else:
        text = (content or "").strip()
    return text or "Не удалось получить ответ от xAI."


# ---------------------------------------------------------------------------
# Telegram I/O helpers
# ---------------------------------------------------------------------------
_MD_PATTERNS = (
    ("***", ""),
    ("**", ""),
    ("__", ""),
    ("```", ""),
)


def strip_markdown(text: str) -> str:
    """Удаляет распространённые markdown-маркеры, чтобы ответ был plain text."""
    out = text
    for src, dst in _MD_PATTERNS:
        out = out.replace(src, dst)
    # одиночные подчёркивания/звёздочки/бэктики на границах слов — убираем,
    # но не трогаем минусы и решётки, чтобы не ломать смысл текста.
    cleaned_lines = []
    for line in out.splitlines():
        stripped = line.lstrip()
        # markdown-заголовки "# ", "## " и т.п. → просто текст
        if stripped.startswith("#"):
            i = 0
            while i < len(stripped) and stripped[i] == "#":
                i += 1
            line = " " * (len(line) - len(stripped)) + stripped[i:].lstrip()
        cleaned_lines.append(line)
    out = "\n".join(cleaned_lines)
    # одиночные * _ ` → убираем
    for ch in ("*", "_", "`"):
        out = out.replace(ch, "")
    return out


def sanitize_telegram_text(text: str) -> str:
    text = strip_markdown(text)
    cleaned = []
    for ch in text:
        code = ord(ch)
        if code in (9, 10, 13) or (32 <= code <= 0xD7FF) or (0xE000 <= code <= 0x10FFFF):
            cleaned.append(ch)
    result = "".join(cleaned).strip()
    return result or "(empty response)"


async def send_reply_safe(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
    if not update.effective_chat:
        return
    chat_id = update.effective_chat.id
    safe_text = sanitize_telegram_text(text)
    chunks = [safe_text[i : i + TG_CHUNK] for i in range(0, len(safe_text), TG_CHUNK)] or [
        "(empty response)"
    ]
    for chunk in chunks:
        try:
            if update.message:
                await update.message.reply_text(chunk)
            else:
                await context.bot.send_message(chat_id=chat_id, text=chunk)
        except TelegramError as exc:
            logger.exception("send failed: %s", exc)
            try:
                await context.bot.send_message(chat_id=chat_id, text=chunk)
            except TelegramError:
                logger.exception("fallback send_message also failed")


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
def get_available_models(provider: str) -> tuple[str, ...]:
    return XAI_MODELS if provider == "xai" else GEMINI_MODELS


def set_runtime_target(application: Application, provider: str, model_name: str) -> None:
    if provider == "xai":
        xai_api_key = os.getenv("XAI_API_KEY")
        if not xai_api_key:
            raise RuntimeError("Не задана переменная окружения: XAI_API_KEY")
        _RUNTIME["xai_api_key"] = xai_api_key
        _RUNTIME.pop("gemini_model", None)
    else:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise RuntimeError("Не задана переменная окружения: GEMINI_API_KEY")
        genai.configure(api_key=gemini_api_key, transport="rest")
        _RUNTIME["gemini_model"] = genai.GenerativeModel(model_name)
    application.bot_data["provider"] = provider
    application.bot_data["model_name"] = model_name


def persist_model(provider: str, model_name: str) -> None:
    set_env_key(ENV_PATH, "MODEL_PROVIDER", provider)
    env_key = "XAI_MODEL" if provider == "xai" else "GEMINI_MODEL"
    set_env_key(ENV_PATH, env_key, model_name)


def resolve_model_selection(raw: str) -> tuple[str, str] | None:
    normalized = " ".join(raw.strip().lower().replace("-", " ").split())
    aliases = {
        "gemini": ("gemini", DEFAULT_GEMINI_MODEL),
        "gemini 3": ("gemini", "gemini-3-flash-preview"),
        "gemini 3 flash": ("gemini", "gemini-3-flash-preview"),
        "gemini 2.5": ("gemini", "gemini-2.5-flash"),
        "gemini 2.5 flash": ("gemini", "gemini-2.5-flash"),
        "gemini 2.5 pro": ("gemini", "gemini-2.5-pro"),
        "grok": ("xai", DEFAULT_XAI_MODEL),
        "grok 4": ("xai", "grok-4-latest"),
        "grok 4 latest": ("xai", "grok-4-latest"),
        "grok 3": ("xai", "grok-3-latest"),
    }
    if normalized in aliases:
        return aliases[normalized]
    if raw in GEMINI_MODELS:
        return ("gemini", raw)
    if raw in XAI_MODELS:
        return ("xai", raw)
    return None


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.chat_data["history"] = []
    provider = context.application.bot_data.get("provider", DEFAULT_PROVIDER)
    model_name = context.application.bot_data.get("model_name", DEFAULT_GEMINI_MODEL)
    await update.message.reply_text(
        "Привет! Я AI-бот.\n"
        f"Провайдер: {provider}\n"
        f"Модель: {model_name}\n\n"
        "Команды: /help"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Команды:\n"
        "/start — приветствие, сброс истории\n"
        "/help — эта справка\n"
        "/ping — проверка живости\n"
        "/model — модели + inline-выбор\n"
        "/model <name> — сменить модель (gemini, grok, gemini 2.5, ...)\n"
        "/temp — текущая температура (0..1)\n"
        "/temp <0..1> — сменить температуру для этого чата\n"
        "/system — показать system prompt\n"
        "/system <text> — установить system prompt\n"
        "/system reset — сбросить system prompt\n"
        "/reset — очистить историю\n"
        "/history — последние сообщения\n"
        "/export — выгрузить историю как JSON-файл\n"
        "/stats — статистика чата\n"
        "/id — id чата и пользователя\n"
        "/whoami — информация о пользователе\n"
        "/broadcast <text> — (admin) рассылка\n"
        "/users — (admin) список чатов\n\n"
        "Также можно прислать фото или голосовое — отвечу через Gemini."
    )


async def ping_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    provider = context.application.bot_data.get("provider", DEFAULT_PROVIDER)
    model_name = context.application.bot_data.get("model_name", DEFAULT_GEMINI_MODEL)
    await update.message.reply_text(f"pong\nprovider={provider}\nmodel={model_name}")


def _build_model_keyboard() -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for m in GEMINI_MODELS:
        rows.append([InlineKeyboardButton(f"Gemini · {m}", callback_data=f"model:gemini:{m}")])
    for m in XAI_MODELS:
        rows.append([InlineKeyboardButton(f"xAI · {m}", callback_data=f"model:xai:{m}")])
    return InlineKeyboardMarkup(rows)


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    provider = context.application.bot_data.get("provider", DEFAULT_PROVIDER)
    current = context.application.bot_data.get("model_name", DEFAULT_GEMINI_MODEL)

    if not context.args:
        await update.message.reply_text(
            f"Провайдер: {provider}\nТекущая модель: {current}\n\nВыбери модель:",
            reply_markup=_build_model_keyboard(),
        )
        return

    requested = " ".join(context.args).strip()
    resolved = resolve_model_selection(requested)
    if not resolved:
        await update.message.reply_text(
            f"Неизвестная модель: {requested}\n"
            f"Gemini: {', '.join(GEMINI_MODELS)}\n"
            f"xAI: {', '.join(XAI_MODELS)}"
        )
        return

    target_provider, target_model = resolved
    try:
        set_runtime_target(context.application, target_provider, target_model)
        persist_model(target_provider, target_model)
        await update.message.reply_text(f"Переключено: {target_provider} / {target_model}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to switch model: %s", exc)
        await update.message.reply_text(
            explain_provider_error(exc, target_provider, target_model)
        )


async def model_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    try:
        _, provider, model_name = query.data.split(":", 2)
    except ValueError:
        await query.edit_message_text("Некорректный выбор.")
        return
    try:
        set_runtime_target(context.application, provider, model_name)
        persist_model(provider, model_name)
        await query.edit_message_text(f"Переключено: {provider} / {model_name}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("model_button failed: %s", exc)
        await query.edit_message_text(explain_provider_error(exc, provider, model_name))


async def temp_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    current = chat_temperature(context)
    if not context.args:
        await update.message.reply_text(
            f"Текущая температура: {current:.2f}\n"
            "Использование: /temp <число от 0 до 1>\nПример: /temp 0.7"
        )
        return
    try:
        parsed = float(context.args[0].replace(",", "."))
    except ValueError:
        await update.message.reply_text("Некорректное значение. Укажи число от 0 до 1.")
        return
    if not (0.0 <= parsed <= 1.0):
        await update.message.reply_text("Температура должна быть в диапазоне 0..1.")
        return
    context.chat_data["temperature"] = parsed
    await update.message.reply_text(f"Температура: {parsed:.2f}")


async def system_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            "System prompt:\n" + chat_system_prompt(context) +
            "\n\n/system <text> — задать\n/system reset — сбросить"
        )
        return
    if context.args[0].lower() == "reset":
        context.chat_data.pop("system_prompt", None)
        await update.message.reply_text("System prompt сброшен на дефолт.")
        return
    new_prompt = " ".join(context.args).strip()
    context.chat_data["system_prompt"] = new_prompt
    await update.message.reply_text("System prompt обновлён.")


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.chat_data["history"] = []
    await update.message.reply_text("История очищена.")


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    history = chat_history(context)
    if not history:
        await update.message.reply_text("История пуста.")
        return
    lines = []
    for item in history[-10:]:
        role = "🧑" if item.get("role") == "user" else "🤖"
        text = (item.get("text") or "").strip().replace("\n", " ")
        if len(text) > 200:
            text = text[:200] + "…"
        lines.append(f"{role} {text}")
    await send_reply_safe(update, context, "\n".join(lines))


async def export_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    history = chat_history(context)
    if not history:
        await update.message.reply_text("История пуста — нечего экспортировать.")
        return
    payload = json.dumps(history, ensure_ascii=False, indent=2).encode("utf-8")
    bio = io.BytesIO(payload)
    bio.name = f"history_{update.effective_chat.id}.json"
    await context.bot.send_document(
        chat_id=update.effective_chat.id,
        document=bio,
        filename=bio.name,
        caption=f"История ({len(history)} сообщений)",
    )


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    stats = context.chat_data.get("stats", {})
    history_len = len(chat_history(context))
    provider = context.application.bot_data.get("provider", DEFAULT_PROVIDER)
    model_name = context.application.bot_data.get("model_name", DEFAULT_GEMINI_MODEL)
    text = (
        f"Статистика чата:\n"
        f"- сообщений в истории: {history_len}\n"
        f"- запросов всего: {stats.get('requests', 0)}\n"
        f"- ответов всего: {stats.get('replies', 0)}\n"
        f"- ошибок: {stats.get('errors', 0)}\n"
        f"- символов в ответах: {stats.get('reply_chars', 0)}\n"
        f"- провайдер: {provider}\n"
        f"- модель: {model_name}\n"
        f"- температура: {chat_temperature(context):.2f}"
    )
    await update.message.reply_text(text)


async def id_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    user = update.effective_user
    await update.message.reply_text(
        f"chat_id: {chat.id if chat else 'n/a'}\n"
        f"chat_type: {chat.type if chat else 'n/a'}\n"
        f"user_id: {user.id if user else 'n/a'}"
    )


async def whoami_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user:
        await update.message.reply_text("Нет данных о пользователе.")
        return
    admin = is_admin(context, user.id)
    username = f"@{user.username}" if user.username else "(не задан)"
    await update.message.reply_text(
        f"id: {user.id}\n"
        f"username: {username}\n"
        f"name: {user.full_name}\n"
        f"admin: {'да' if admin else 'нет'}"
    )


async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not is_admin(context, user.id):
        await update.message.reply_text("Команда только для админов.")
        return
    if not context.args:
        await update.message.reply_text("Использование: /broadcast <текст>")
        return
    text = " ".join(context.args)
    chats: dict[int, dict[str, Any]] = context.application.bot_data.get("known_chats", {})
    sent = 0
    failed = 0
    for chat_id in list(chats.keys()):
        try:
            await context.bot.send_message(chat_id=chat_id, text=text)
            sent += 1
        except TelegramError:
            failed += 1
    await update.message.reply_text(f"Рассылка завершена. Отправлено: {sent}, ошибок: {failed}.")


async def users_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not is_admin(context, user.id):
        await update.message.reply_text("Команда только для админов.")
        return
    chats: dict[int, dict[str, Any]] = context.application.bot_data.get("known_chats", {})
    if not chats:
        await update.message.reply_text("Нет известных чатов.")
        return
    lines = [f"{cid}: {info.get('title', '')}" for cid, info in chats.items()]
    await send_reply_safe(update, context, "Известные чаты:\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# Message handlers
# ---------------------------------------------------------------------------
async def _pre_check(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    chat = update.effective_chat
    user = update.effective_user
    if not chat or not user:
        return False
    if not is_chat_allowed(context, chat.id):
        await send_reply_safe(update, context, "Этот чат не в списке разрешённых.")
        return False
    wait = cooldown_check(context, user.id)
    if wait > 0:
        await send_reply_safe(update, context, f"Слишком часто. Подожди {wait:.1f} с.")
        return False
    remember_chat(context, chat.id, chat.title or chat.full_name or str(chat.id))
    return True


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    if not await _pre_check(update, context):
        return

    user_text = update.message.text.strip()
    if not user_text:
        await update.message.reply_text("Пожалуйста, отправьте текстовое сообщение.")
        return

    provider = context.application.bot_data.get("provider", DEFAULT_PROVIDER)
    model_name = context.application.bot_data.get("model_name", DEFAULT_GEMINI_MODEL)
    temperature = chat_temperature(context)
    prompt = build_prompt(context, user_text)

    bump_stats(context, "requests")

    try:
        if provider == "xai":
            api_key = _RUNTIME["xai_api_key"]
            text = await asyncio.to_thread(
                generate_with_xai, api_key, model_name, prompt, temperature
            )
        else:
            model = _RUNTIME["gemini_model"]
            text = await asyncio.to_thread(generate_with_gemini, model, prompt, temperature, None)

        append_history(context, "user", user_text)
        append_history(context, "assistant", text)
        bump_stats(context, "replies")
        bump_stats(context, "reply_chars", len(text))
        await send_reply_safe(update, context, text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error while calling %s: %s", provider, exc)
        bump_stats(context, "errors")
        await send_reply_safe(update, context, explain_provider_error(exc, provider, model_name))


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    voice = update.message.voice or update.message.audio
    if not voice:
        return
    if not await _pre_check(update, context):
        return

    provider = context.application.bot_data.get("provider", DEFAULT_PROVIDER)
    if provider != "gemini":
        await update.message.reply_text(
            "Голосовые поддерживаются только Gemini. Переключи: /model gemini"
        )
        return

    bump_stats(context, "requests")
    await typing(context, update.effective_chat.id)

    try:
        tg_file = await context.bot.get_file(voice.file_id)
        buf = io.BytesIO()
        await tg_file.download_to_memory(out=buf)
        data = buf.getvalue()
        mime = getattr(voice, "mime_type", None) or "audio/ogg"

        caption = (update.message.caption or "").strip()
        user_part = (
            f"[Пользователь прислал голосовое сообщение] {caption}".strip()
            if caption
            else "[Пользователь прислал голосовое сообщение] "
                 "Расшифруй его и ответь по сути."
        )
        prompt = build_prompt(context, user_part)

        model = _RUNTIME["gemini_model"]
        text = await asyncio.to_thread(
            generate_with_gemini,
            model,
            prompt,
            chat_temperature(context),
            None,
            (mime, data),
        )

        append_history(context, "user", f"[voice {len(data)}b {mime}] {caption}".strip())
        append_history(context, "assistant", text)
        bump_stats(context, "replies")
        bump_stats(context, "reply_chars", len(text))
        await send_reply_safe(update, context, text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Voice handling failed: %s", exc)
        bump_stats(context, "errors")
        await send_reply_safe(
            update,
            context,
            explain_provider_error(
                exc, provider, context.application.bot_data.get("model_name", "")
            ),
        )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.photo:
        return
    if not await _pre_check(update, context):
        return

    provider = context.application.bot_data.get("provider", DEFAULT_PROVIDER)
    if provider != "gemini":
        await update.message.reply_text(
            "Анализ фото поддерживается только Gemini. Переключи: /model gemini"
        )
        return

    caption = (update.message.caption or "Опиши, что на фото, и помоги пользователю.").strip()
    photo = update.message.photo[-1]

    bump_stats(context, "requests")

    try:
        tg_file = await context.bot.get_file(photo.file_id)
        buf = io.BytesIO()
        await tg_file.download_to_memory(out=buf)
        buf.seek(0)
        image = Image.open(buf)
        image.load()

        model = _RUNTIME["gemini_model"]
        prompt = build_prompt(context, f"[Пользователь прислал фото] {caption}")
        text = await asyncio.to_thread(
            generate_with_gemini, model, prompt, chat_temperature(context), image
        )

        append_history(context, "user", f"[photo] {caption}")
        append_history(context, "assistant", text)
        bump_stats(context, "replies")
        bump_stats(context, "reply_chars", len(text))
        await send_reply_safe(update, context, text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Photo handling failed: %s", exc)
        bump_stats(context, "errors")
        await send_reply_safe(
            update,
            context,
            explain_provider_error(
                exc, provider, context.application.bot_data.get("model_name", "")
            ),
        )


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled error: %s", context.error)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
async def _post_init(application: Application) -> None:
    from telegram import BotCommand

    # Runtime objects (модель Gemini, api keys) и настройки из .env.
    # Делается ЗДЕСЬ, а не в main(), потому что persistence затирает bot_data
    # своими сохранёнными значениями уже после Application.builder().build().
    provider = os.getenv("MODEL_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    if provider == "xai":
        _RUNTIME["xai_api_key"] = get_required_env("XAI_API_KEY")
        model_name = os.getenv("XAI_MODEL", DEFAULT_XAI_MODEL)
    else:
        gemini_api_key = get_required_env("GEMINI_API_KEY")
        model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        genai.configure(api_key=gemini_api_key, transport="rest")
        _RUNTIME["gemini_model"] = genai.GenerativeModel(model_name)

    application.bot_data["provider"] = provider
    application.bot_data["model_name"] = model_name
    application.bot_data["temperature"] = parse_temperature(os.getenv("AI_TEMPERATURE"))
    application.bot_data["admin_ids"] = parse_int_list(os.getenv("TELEGRAM_ADMIN_IDS"))
    application.bot_data["allowed_chats"] = parse_int_list(os.getenv("ALLOWED_CHAT_IDS"))
    try:
        application.bot_data["cooldown"] = float(
            os.getenv("USER_COOLDOWN_SECONDS", str(DEFAULT_COOLDOWN))
        )
    except ValueError:
        application.bot_data["cooldown"] = DEFAULT_COOLDOWN

    commands = [
        BotCommand("start", "Приветствие, сброс истории"),
        BotCommand("help", "Список команд"),
        BotCommand("model", "Сменить/выбрать модель"),
        BotCommand("temp", "Температура генерации"),
        BotCommand("system", "System prompt"),
        BotCommand("reset", "Очистить историю"),
        BotCommand("history", "Последние сообщения"),
        BotCommand("export", "Выгрузить историю JSON"),
        BotCommand("stats", "Статистика чата"),
        BotCommand("id", "ID чата и пользователя"),
        BotCommand("whoami", "Кто я"),
        BotCommand("ping", "Проверка живости"),
    ]
    try:
        await application.bot.set_my_commands(commands)
    except TelegramError:
        logger.exception("set_my_commands failed")


def main() -> None:
    load_env_file()

    telegram_token = get_required_env("TELEGRAM_BOT_TOKEN")

    application = (
        Application.builder()
        .token(telegram_token)
        .post_init(_post_init)
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("ping", ping_command))
    application.add_handler(CommandHandler("model", model_command))
    application.add_handler(CommandHandler("temp", temp_command))
    application.add_handler(CommandHandler("system", system_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(CommandHandler("history", history_command))
    application.add_handler(CommandHandler("export", export_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("id", id_command))
    application.add_handler(CommandHandler("whoami", whoami_command))
    application.add_handler(CommandHandler("broadcast", broadcast_command))
    application.add_handler(CommandHandler("users", users_command))
    application.add_handler(CallbackQueryHandler(model_button, pattern=r"^model:"))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(on_error)

    logger.info("Bot starting…")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
