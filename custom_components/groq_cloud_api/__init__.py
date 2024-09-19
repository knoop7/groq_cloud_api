"""The Groq Cloud API integration."""
from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady

from .const import DOMAIN, LOGGER

PLATFORMS: list[Platform] = [Platform.CONVERSATION]

class GroqConfigEntry:
    """Groq Cloud API configuration entry."""

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry):
        """Initialize the config entry."""
        self.hass = hass
        self.config_entry = config_entry
        self.runtime_data = config_entry.data[CONF_API_KEY]
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
    except Exception as ex:
        LOGGER.error("Error setting up Groq Cloud API: %s", ex)
        raise ConfigEntryNotReady from ex

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    if unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        hass.data[DOMAIN].pop(entry.entry_id, None)

    return unload_ok
