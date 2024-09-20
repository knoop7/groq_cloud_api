from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady

from .const import DOMAIN, LOGGER, CONF_PROXY

PLATFORMS: list[Platform] = [Platform.CONVERSATION]

class GroqConfigEntry:
    """Groq Cloud API configuration entry."""

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry):
        """Initialize the config entry."""
        self.hass = hass
        self.config_entry = config_entry
        self.api_key = config_entry.data[CONF_API_KEY]
        self.proxy = config_entry.data.get(CONF_PROXY)
        self.options = config_entry.options

    @property
    def entry_id(self):
        """Return the entry ID."""
        return self.config_entry.entry_id

    @property
    def title(self):
        """Return the title of the config entry."""
        return self.config_entry.title

    def async_on_unload(self, func):
        """Add a function to call when config entry is unloaded."""
        return self.config_entry.async_on_unload(func)

    def async_add_update_listener(self, listener):
        """Add a listener for when the config entry is updated."""
        return self.config_entry.async_add_update_listener(listener)

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Groq Cloud API from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    try:
        groq_entry = GroqConfigEntry(hass, entry)
        hass.data[DOMAIN][entry.entry_id] = groq_entry
        LOGGER.info("成功设置 Groq Cloud API，条目 ID: %s", entry.entry_id)
        
        if groq_entry.proxy:
            LOGGER.info("Groq Cloud API 已配置代理: %s", groq_entry.proxy)
        else:
            LOGGER.info("Groq Cloud API 未配置代理")

    except Exception as ex:
        LOGGER.error("设置 Groq Cloud API 时出错: %s", ex)
        raise ConfigEntryNotReady from ex

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    if unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        hass.data[DOMAIN].pop(entry.entry_id, None)
        LOGGER.info("已卸载 Groq Cloud API 条目，ID: %s", entry.entry_id)

    return unload_ok
