# AI Telegram Bot (Gemini + xAI Grok)

Telegram-бот с поддержкой двух провайдеров (Google Gemini и xAI Grok), историей диалога,
персистентностью, vision (фото через Gemini), админ-командами и rate-limit'ом.

## Возможности

- Два провайдера: `gemini` и `xai`, переключение на лету (`/model`).
- Inline-клавиатура для выбора модели.
- История диалога per-chat (последние N сообщений), переживает рестарт (`PicklePersistence`).
- Кастомный system prompt per-chat (`/system`).
- Регулировка температуры per-chat (`/temp`).
- Поддержка фото через Gemini vision.
- Typing-индикатор во время генерации.
- Rate-limit на пользователя.
- Whitelist чатов и админ-команды (`/broadcast`, `/users`).
- Экспорт истории в JSON (`/export`).
- Статистика (`/stats`).

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # затем заполни ключи
python bot.py
```

## Команды

| Команда | Описание |
|---|---|
| `/start` | Приветствие, сброс истории |
| `/help` | Справка |
| `/ping` | Проверка живости + текущая модель |
| `/model` | Показать модели + inline-выбор |
| `/model <name>` | Сменить модель (`gemini`, `grok`, и т.п.) |
| `/temp` | Текущая температура (0..1) |
| `/temp <0..1>` | Сменить температуру |
| `/system` | Текущий system prompt |
| `/system <text>` | Установить system prompt для этого чата |
| `/system reset` | Сбросить к дефолтному |
| `/reset` | Очистить историю диалога |
| `/history` | Показать последние сообщения |
| `/export` | Выгрузить историю как JSON-файл |
| `/stats` | Статистика чата |
| `/id` | ID чата и пользователя |
| `/whoami` | Информация о пользователе |
| `/broadcast <text>` | (admin) рассылка по всем чатам |
| `/users` | (admin) список известных чатов |

## ENV

См. `.env.example`. Минимум: `TELEGRAM_BOT_TOKEN` + ключ выбранного провайдера.

## Лицензия

MIT
