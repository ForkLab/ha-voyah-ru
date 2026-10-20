"""Switch platform for the Voyah integration (remote climate on the new telemetry API)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from homeassistant.components.switch import SwitchEntity, SwitchEntityDescription
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .api import VoyahApiError
from .const import COMMAND_REFRESH_DELAY_SECONDS, CONF_CAR_ID, CONF_CAR_NAME, DOMAIN
from .coordinator import VoyahDataUpdateCoordinator


@dataclass(frozen=True, kw_only=True)
class VoyahSwitchEntityDescription(SwitchEntityDescription):
    """Description of a switch backed by a pair of telemetry commands."""

    # activateCommand of the telemetry button; its status is the switch state
    turn_on_command: str
    # deactivateCommand of the same button
    turn_off_command: str


SWITCH_DESCRIPTIONS: tuple[VoyahSwitchEntityDescription, ...] = (
    VoyahSwitchEntityDescription(
        key="heating",
        translation_key="heating",
        icon="mdi:radiator",
        turn_on_command="heatingOn",
        turn_off_command="heatingOff",
    ),
    VoyahSwitchEntityDescription(
        key="cooling",
        translation_key="cooling",
        icon="mdi:snowflake",
        turn_on_command="coolingOn",
        turn_off_command="coolingOff",
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Voyah switches for commands the new telemetry API reports."""
    coordinator: VoyahDataUpdateCoordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    command_states = coordinator.data.get("command_states", {})
    async_add_entities(
        VoyahCommandSwitch(coordinator, description, entry)
        for description in SWITCH_DESCRIPTIONS
        if description.turn_on_command in command_states
    )


class VoyahCommandSwitch(CoordinatorEntity[VoyahDataUpdateCoordinator], SwitchEntity):
    """Switch that sends on/off commands and shows the state reported by the API."""

    _attr_has_entity_name = True
    entity_description: VoyahSwitchEntityDescription

    def __init__(
        self,
        coordinator: VoyahDataUpdateCoordinator,
        description: VoyahSwitchEntityDescription,
        entry: ConfigEntry,
    ) -> None:
        super().__init__(coordinator)
        self.entity_description = description
        car_id = entry.data.get(CONF_CAR_ID, entry.entry_id)
        self._attr_unique_id = f"{car_id}_{description.key}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, car_id)},
            name=entry.data.get(CONF_CAR_NAME, "Voyah"),
            manufacturer="Voyah",
        )

    @property
    def is_on(self) -> bool | None:
        """Return the button status reported by the telemetry API."""
        command_states = self.coordinator.data.get("command_states", {})
        return command_states.get(self.entity_description.turn_on_command)

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Send the activate command."""
        await self._async_send(self.entity_description.turn_on_command)

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Send the deactivate command."""
        await self._async_send(self.entity_description.turn_off_command)

    async def _async_send(self, command: str) -> None:
        try:
            await self.coordinator.client.async_send_command(None, command)
        except VoyahApiError as err:
            raise HomeAssistantError(f"Voyah command failed: {err}") from err

        if self.hass is None:
            await self.coordinator.async_request_refresh()
            return

        for delay in COMMAND_REFRESH_DELAY_SECONDS:
            async_call_later(self.hass, delay, self._async_delayed_refresh)

    @callback
    def _async_delayed_refresh(self, _now: datetime) -> None:
        """Request a coordinator refresh after the car has time to apply the command."""
        if self.hass is not None:
            self.hass.async_create_task(self.coordinator.async_request_refresh())
