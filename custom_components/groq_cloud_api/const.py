import logging

DOMAIN = "groq_cloud_api"

LOGGER = logging.getLogger(__name__)

DEFAULT_NAME = "Groq Cloud API"
CONF_RECOMMENDED = "recommended"
CONF_PROMPT = "prompt"
CONF_CHAT_MODEL = "chat_model"
RECOMMENDED_CHAT_MODEL = "llama-3.1-70b-versatile"
CONF_MAX_TOKENS = "max_tokens"
RECOMMENDED_MAX_TOKENS = 150
CONF_TOP_P = "top_p"
RECOMMENDED_TOP_P = 1.0
CONF_TEMPERATURE = "temperature"
RECOMMENDED_TEMPERATURE = 1.0
CONF_MAX_HISTORY_MESSAGES = "max_history_messages"
RECOMMENDED_MAX_HISTORY_MESSAGES = 5
CONF_PROXY = "代理服务器地址（可选)"
