import aiohttp
from aiohttp_proxy import ProxyConnector, ProxyType
import json
from typing import Any, Literal, TypedDict

from voluptuous_openapi import convert
import voluptuous as vol

from homeassistant.components import assist_pipeline, conversation
from homeassistant.components.conversation import trace
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, TemplateError
from homeassistant.helpers import device_registry as dr, intent, llm, template
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import ulid

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_MAX_HISTORY_MESSAGES, 
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_MAX_HISTORY_MESSAGES,  
    RECOMMENDED_TOP_P,
    CONF_PROXY,
)

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

class ChatCompletionMessageParam(TypedDict, total=False):
    role: str
    content: str | None
    name: str | None
    tool_calls: list["ChatCompletionMessageToolCallParam"] | None

class Function(TypedDict, total=False):
    name: str
    arguments: str

class ChatCompletionMessageToolCallParam(TypedDict):
    id: str
    type: str
    function: Function

class ChatCompletionToolParam(TypedDict):
    type: str
    function: dict[str, Any]

def _format_tool(
    tool: llm.Tool, custom_serializer: Any | None
) -> ChatCompletionToolParam:
    """Format tool specification."""
    tool_spec = {
        "name": tool.name,
        "parameters": convert(tool.parameters, custom_serializer=custom_serializer),
    }
    if tool.description:
        tool_spec["description"] = tool.description
    return ChatCompletionToolParam(type="function", function=tool_spec)

class GroqConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """Groq conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = entry
        self.history: dict[str, list[ChatCompletionMessageParam]] = {}
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Groq",
            model="Groq Cloud",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        if self.entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = conversation.ConversationEntityFeature.CONTROL
        self.http_proxy = entry.data.get(CONF_PROXY)
    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        assist_pipeline.async_migrate_engine(
            self.hass, "conversation", self.entry.entry_id, self.entity_id
        )
        conversation.async_set_agent(self.hass, self.entry, self)
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    async def async_will_remove_from_hass(self) -> None:
        """When entity is removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence."""
        options = self.entry.options
        intent_response = intent.IntentResponse(language=user_input.language)
        llm_api: llm.APIInstance | None = None
        tools: list[ChatCompletionToolParam] | None = None
        user_name: str | None = None
        llm_context = llm.LLMContext(
            platform=DOMAIN,
            context=user_input.context,
            user_prompt=user_input.text,
            language=user_input.language,
            assistant=conversation.DOMAIN,
            device_id=user_input.device_id,
        )

        if options.get(CONF_LLM_HASS_API):
            try:
                llm_api = await llm.async_get_api(
                    self.hass,
                    options[CONF_LLM_HASS_API],
                    llm_context,
                )
            except HomeAssistantError as err:
                LOGGER.error("获取 LLM API 时出错：%s", err)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"准备 LLM API 时出错： {err}",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=user_input.conversation_id
                )
            tools = [
                _format_tool(tool, llm_api.custom_serializer) for tool in llm_api.tools
            ]

        if user_input.conversation_id is None:
            conversation_id = ulid.ulid_now()
            messages = []
        elif user_input.conversation_id in self.history:
            conversation_id = user_input.conversation_id
            messages = self.history[conversation_id]
        else:
            conversation_id = user_input.conversation_id
            messages = []

        max_history_messages = options.get(CONF_MAX_HISTORY_MESSAGES, RECOMMENDED_MAX_HISTORY_MESSAGES)
        use_history = len(messages) < max_history_messages

        if (
            user_input.context
            and user_input.context.user_id
            and (
                user := await self.hass.auth.async_get_user(user_input.context.user_id)
            )
        ):
            user_name = user.name

        try:
            prompt_parts = [
                template.Template(
                    llm.BASE_PROMPT
                    + options.get(CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT),
                    self.hass,
                ).async_render(
                    {
                        "ha_name": self.hass.config.location_name,
                        "user_name": user_name,
                        "llm_context": llm_context,
                    },
                    parse_result=False,
                )
            ]
        except TemplateError as err:
            LOGGER.error("Error rendering prompt: %s", err)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"抱歉，我的模板有问题： {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        if llm_api:
            prompt_parts.append(llm_api.api_prompt)

        prompt = "\n".join(prompt_parts)

        messages = [
            ChatCompletionMessageParam(role="system", content=prompt),
            *(messages if use_history else []),
            ChatCompletionMessageParam(role="user", content=user_input.text),
        ]
        max_history_messages = options.get(CONF_MAX_HISTORY_MESSAGES, RECOMMENDED_MAX_HISTORY_MESSAGES)
        if len(messages) > max_history_messages + 1:  # +1 是因为系统提示消息
            messages = [messages[0]] + messages[-(max_history_messages):]
            
        LOGGER.debug("Prompt: %s", messages)
        LOGGER.debug("Tools: %s", tools)
        trace.async_conversation_trace_append(
            trace.ConversationTraceEventType.AGENT_DETAIL,
            {"messages": messages, "tools": llm_api.tools if llm_api else None},
        )

        api_key = self.entry.data[CONF_API_KEY]
        try:
            # 创建代理连接器
            if self.http_proxy:
                connector = ProxyConnector.from_url(self.http_proxy)
            else:
                connector = None

            async with aiohttp.ClientSession(connector=connector) as session:
                for _iteration in range(MAX_TOOL_ITERATIONS):
                    try:
                        payload = {
                            "model": options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
                            "messages": messages,
                            "max_tokens": options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                            "top_p": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                            "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                            "user": conversation_id,
                        }
                        if tools:
                            payload["tools"] = tools

                        async with session.post(
                            GROQ_API_URL,
                            json=payload,
                            headers={"Authorization": f"Bearer {api_key}"}
                        ) as response:
                            if response.status != 200:
                                raise HomeAssistantError(f"Groq API 返回状态 {response.status}")
                            result = await response.json()
                    except Exception as err:
                        raise HomeAssistantError(f"与 Groq API 通信时出错: {err}")

                    LOGGER.debug("Response %s", result)
                    response = result["choices"][0]["message"]

                    messages.append(response)
                    tool_calls = response.get("tool_calls")

                    if not tool_calls or not llm_api:
                        break

                    for tool_call in tool_calls:
                        tool_input = llm.ToolInput(
                            tool_name=tool_call["function"]["name"],
                            tool_args=json.loads(tool_call["function"]["arguments"]),
                        )
                        LOGGER.debug(
                            "Tool call: %s(%s)", tool_input.tool_name, tool_input.tool_args
                        )

                        try:
                            tool_response = await llm_api.async_call_tool(tool_input)
                        except (HomeAssistantError, vol.Invalid) as e:
                            tool_response = {"error": type(e).__name__}
                            if str(e):
                                tool_response["error_text"] = str(e)

                        LOGGER.debug("Tool response: %s", tool_response)
                        messages.append(
                            ChatCompletionMessageParam(
                                role="tool",
                                tool_call_id=tool_call["id"],
                                content=json.dumps(tool_response),
                            )
                        )

        except Exception as err:
            LOGGER.error("处理 Groq 请求时出错: %s", err)
            
            # 尝试使用内置的 Home Assistant LLM
            try:
                hass_llm = await llm.async_get_assistant(self.hass, llm_context)
                result = await hass_llm.async_process(user_input.text)
                intent_response.async_set_speech(result.response)
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=conversation_id
                )
            except Exception as hass_llm_err:
                LOGGER.error("使用 Home Assistant LLM 时出错: %s", hass_llm_err)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"已执行成功，如果失败请重试。",
                )
            
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        if use_history:
            self.history[conversation_id] = messages

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response.get("content") or "")
        return conversation.ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        await hass.config_entries.async_reload(entry.entry_id)

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Groq conversation platform."""
    async_add_entities([GroqConversationEntity(config_entry)])

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    if unload_ok := await hass.config_entries.async_unload_platforms(entry, ["conversation"]):
        hass.data[DOMAIN].pop(entry.entry_id, None)
    return unload_ok
