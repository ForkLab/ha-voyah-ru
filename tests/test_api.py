"""Tests for Voyah API client."""

import base64
from datetime import UTC, datetime
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import time_machine

from custom_components.voyah.api import (
    VoyahApiAuthError,
    VoyahApiClient,
    VoyahApiConnectionError,
    VoyahApiError,
    _user_id_from_token,
)
from custom_components.voyah.const import API_BASE_URL

from .conftest import (
    MOCK_ACCESS_TOKEN,
    MOCK_CAR_ID,
    MOCK_CAR_INFO_DATA,
    MOCK_IMEI,
    MOCK_REFRESH_TOKEN,
    MOCK_TELEMETRY_RESPONSE,
    MOCK_VIN,
)

LEGACY_SENSORS_RESPONSE = {
    "sensorsData": {"batteryPercentage": 80},
    "positionData": {"lat": 55.0, "lon": 37.0},
    "time": 123,
}


def _make_client(session: MagicMock, access_token: str = MOCK_ACCESS_TOKEN) -> VoyahApiClient:
    return VoyahApiClient(session, MOCK_CAR_ID, access_token, MOCK_REFRESH_TOKEN)


def _mock_response(status: int, json_data: dict) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data)
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def _routed_session(routes: dict[str, MagicMock]) -> MagicMock:
    """Session whose request() answers by URL path prefix."""

    def _request(method: str, url: str, **kwargs) -> MagicMock:
        path = url.removeprefix(API_BASE_URL)
        for prefix, response in routes.items():
            if path.startswith(prefix):
                return response
        raise AssertionError(f"Unexpected request: {method} {url}")

    session = MagicMock()
    session.request = MagicMock(side_effect=_request)
    return session


def _requested_paths(session: MagicMock) -> list[str]:
    return [call.args[1].removeprefix(API_BASE_URL) for call in session.request.call_args_list]


def _fake_jwt(claims: dict) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"header.{payload}.signature"


def _new_telemetry_routes() -> dict[str, MagicMock]:
    return {
        f"/car-service/car/v2/{MOCK_CAR_ID}": _mock_response(200, MOCK_CAR_INFO_DATA),
        "/config-service/config/flags": _mock_response(200, {"flags": {"newTelemetryEnabled": True}}),
        f"/client-bff-service/telemetry/{MOCK_IMEI}": _mock_response(200, MOCK_TELEMETRY_RESPONSE),
        "/client-bff-service/telemetry-commands": _mock_response(201, {"commandId": "cmd-1"}),
    }


def _legacy_routes(car_info: dict = MOCK_CAR_INFO_DATA) -> dict[str, MagicMock]:
    return {
        f"/car-service/car/v2/{MOCK_CAR_ID}": _mock_response(200, car_info),
        "/config-service/config/flags": _mock_response(200, {"flags": {"newTelemetryEnabled": False}}),
        f"/car-service/tbox/{MOCK_CAR_ID}/sensors": _mock_response(200, LEGACY_SENSORS_RESPONSE),
        f"/car-service/tbox/{MOCK_CAR_ID}/": _mock_response(200, {}),
    }


# ── Legacy TBox response parsing ─────────────────────────────────────────────


async def test_parse_extracts_fields() -> None:
    """_parse pulls sensorsData, positionData and time."""
    raw = {
        "sensorsData": {"batteryPercentage": 75},
        "positionData": {"lat": 55.0, "lon": 37.0, "speed": 0},
        "time": 1700000000,
    }
    result = VoyahApiClient._parse(raw)
    assert result["sensors_data"]["batteryPercentage"] == 75
    assert result["position_data"]["lat"] == 55.0
    assert result["time"] == 1700000000


async def test_parse_merges_speed_from_position() -> None:
    """Speed from positionData is merged into sensors_data."""
    raw = {
        "sensorsData": {},
        "positionData": {"speed": 60},
        "time": None,
    }
    result = VoyahApiClient._parse(raw)
    assert result["sensors_data"]["speed"] == 60


async def test_parse_accepts_info_response_wrapper() -> None:
    """_parse accepts the /info response shape with sensors wrapper."""
    raw = {
        "sensors": {
            "sensorsData": {"centralLockingStatus": 1},
            "positionData": {"speed": 0},
            "time": 1784903300,
            "lastPing": 12.5,
        },
        "isCentralLockingOn": False,
    }
    result = VoyahApiClient._parse(raw)
    assert result["sensors_data"]["centralLockingStatus"] == 1
    assert result["time"] == 1784903300
    assert result["last_ping"] == 12.5


# ── New telemetry response parsing ───────────────────────────────────────────


async def test_parse_telemetry_converts_string_values() -> None:
    """String values become numbers/booleans, empty strings become None."""
    sensors = VoyahApiClient._parse_telemetry(MOCK_TELEMETRY_RESPONSE)["sensors_data"]

    assert sensors["batteryPercentage"] == 91
    assert sensors["odometer"] == 14348
    assert sensors["outsideTemp"] == -1.5
    assert sensors["12VBatteryVoltage"] == pytest.approx(14.54)
    assert sensors["tirePressureFL"] == 2.1
    assert sensors["headLightsStatus"] == 0
    assert sensors["batteryTemp"] is None
    assert sensors["doorRRStatus"] is True


async def test_parse_telemetry_keeps_non_numeric_strings() -> None:
    """Non-numeric strings are kept trimmed; decimal commas and TRUE/FALSE are understood."""
    raw = {"sensors": {"gear": " P ", "voltage": "12,5", "flag": "TRUE"}}
    sensors = VoyahApiClient._parse_telemetry(raw)["sensors_data"]
    assert sensors == {"gear": "P", "voltage": 12.5, "flag": True}


async def test_parse_telemetry_maps_keys_to_existing_entities() -> None:
    """Renamed API keys are exposed under the keys existing entities use."""
    sensors = VoyahApiClient._parse_telemetry(MOCK_TELEMETRY_RESPONSE)["sensors_data"]

    assert sensors["remainsMileage"] == 113
    assert sensors["remainsMileageFuel"] == 563
    assert sensors["ignitionStatus"] is True
    # Locked car → lock binary sensor is off (HA: on means unlocked).
    assert sensors["centralLockingStatus"] is False
    # Gun inserted and 12V bus powered → charging, as in the web app.
    assert sensors["chargingStatus"] is True


async def test_parse_telemetry_unlocked_car() -> None:
    """isCentralLockingOn=false maps to an unlocked (on) lock sensor."""
    raw = {**MOCK_TELEMETRY_RESPONSE, "sensors": {"isCentralLockingOn": "false"}}
    sensors = VoyahApiClient._parse_telemetry(raw)["sensors_data"]
    assert sensors["centralLockingStatus"] is True


async def test_parse_telemetry_not_charging_when_12v_low() -> None:
    """Inserted gun alone is not charging when the 12V bus is not powered."""
    raw = {
        **MOCK_TELEMETRY_RESPONSE,
        "sensors": {"isChargingGunInserted": True, "12VBatteryVoltage": "12.4"},
    }
    sensors = VoyahApiClient._parse_telemetry(raw)["sensors_data"]
    assert sensors["chargingStatus"] is False


async def test_parse_telemetry_position_time_and_last_ping() -> None:
    """GPS goes to position_data; time and last_ping come from carStateUpdatedAt."""
    now = datetime.fromtimestamp(1792471790.719 + 30, tz=UTC)
    with time_machine.travel(now, tick=False):
        result = VoyahApiClient._parse_telemetry(MOCK_TELEMETRY_RESPONSE)

    assert result["position_data"] == {
        "lat": 55.75,
        "lon": 37.61,
        "course": 258,
        "position_at": pytest.approx(1792471790.709),
    }
    assert "lat" not in result["sensors_data"]
    assert result["time"] == pytest.approx(1792471790.719)
    assert result["last_ping"] == pytest.approx(30)


async def test_parse_telemetry_collects_command_states() -> None:
    """Button statuses are keyed by their activate command."""
    states = VoyahApiClient._parse_telemetry(MOCK_TELEMETRY_RESPONSE)["command_states"]
    assert states == {
        "heatingOn": True,
        "coolingOn": False,
        "trunkOpen": False,
        "centralLockingOff": False,
        "search": False,
    }


async def test_parse_telemetry_collects_checks() -> None:
    """Firmware and settings checks are passed through for diagnostic sensors."""
    checks = VoyahApiClient._parse_telemetry(MOCK_TELEMETRY_RESPONSE)["checks"]
    assert checks == {
        "firmwareCheck": {"expected": "fw-1.0", "actual": "fw-1.1", "mismatch": True},
        "settingsCheck": {"expected": "cfg-1", "actual": "cfg-1", "mismatch": False},
    }


async def test_parse_telemetry_online_from_top_level() -> None:
    """Online state comes from the top-level isOnline, as in the web app."""
    raw = {**MOCK_TELEMETRY_RESPONSE, "isOnline": False}
    sensors = VoyahApiClient._parse_telemetry(raw)["sensors_data"]
    assert sensors["isOnline"] is False


async def test_parse_telemetry_handles_empty_response() -> None:
    """Missing sections produce empty data instead of errors."""
    result = VoyahApiClient._parse_telemetry({})
    assert result == {
        "sensors_data": {},
        "position_data": {},
        "time": None,
        "last_ping": None,
        "command_states": {},
        "checks": {},
    }


# ── Telemetry API selection ──────────────────────────────────────────────────


async def test_get_car_data_uses_new_telemetry_when_flag_enabled() -> None:
    """Cars with newTelemetryEnabled are polled via client-bff-service by IMEI."""
    session = _routed_session(_new_telemetry_routes())
    client = _make_client(session, access_token=_fake_jwt({"_id": "user-1"}))

    data = await client.async_get_car_data()

    assert data["sensors_data"]["batteryPercentage"] == 91
    assert _requested_paths(session) == [
        f"/car-service/car/v2/{MOCK_CAR_ID}",
        "/config-service/config/flags",
        f"/client-bff-service/telemetry/{MOCK_IMEI}",
    ]
    flags_call = session.request.call_args_list[1]
    assert flags_call.kwargs["params"] == {
        "brand": "voyah",
        "modification": "H56b",
        "userId": "user-1",
        "vin": MOCK_VIN,
    }


async def test_get_car_data_detects_api_once() -> None:
    """Once the new telemetry API is detected, polls go straight to it."""
    session = _routed_session(_new_telemetry_routes())
    client = _make_client(session)

    await client.async_get_car_data()
    session.request.reset_mock()
    await client.async_get_car_data()

    assert _requested_paths(session) == [f"/client-bff-service/telemetry/{MOCK_IMEI}"]


async def test_get_car_data_uses_legacy_api_when_flag_disabled() -> None:
    """Cars without the flag keep using the TBox sensors endpoint."""
    session = _routed_session(_legacy_routes())
    client = _make_client(session)

    data = await client.async_get_car_data()

    assert data["sensors_data"]["batteryPercentage"] == 80
    assert _requested_paths(session)[-1] == f"/car-service/tbox/{MOCK_CAR_ID}/sensors"


async def test_get_car_data_rechecks_flag_periodically() -> None:
    """Legacy cars re-check the rollout flag only after the recheck interval."""
    session = _routed_session(_legacy_routes())
    client = _make_client(session)

    with patch("custom_components.voyah.api.time.monotonic", return_value=1000.0):
        await client.async_get_car_data()
        session.request.reset_mock()
        await client.async_get_car_data()
    assert _requested_paths(session) == [f"/car-service/tbox/{MOCK_CAR_ID}/sensors"]

    session.request.reset_mock()
    with patch("custom_components.voyah.api.time.monotonic", return_value=1000.0 + 13 * 3600):
        await client.async_get_car_data()
    assert "/config-service/config/flags" in _requested_paths(session)


async def test_get_car_data_skips_flags_without_imei() -> None:
    """A car without IMEI cannot use the new API, so flags are not requested."""
    car_info = {key: value for key, value in MOCK_CAR_INFO_DATA.items() if key != "imei"}
    session = _routed_session(_legacy_routes(car_info))
    client = _make_client(session)

    await client.async_get_car_data()

    assert _requested_paths(session) == [
        f"/car-service/car/v2/{MOCK_CAR_ID}",
        f"/car-service/tbox/{MOCK_CAR_ID}/sensors",
    ]


async def test_get_car_data_falls_back_to_legacy_when_detection_fails() -> None:
    """If the flags service fails, poll the legacy API and retry detection next time."""
    routes = _legacy_routes()
    routes["/config-service/config/flags"] = _mock_response(500, {})
    session = _routed_session(routes)
    client = _make_client(session)

    data = await client.async_get_car_data()
    assert data["sensors_data"]["batteryPercentage"] == 80

    session.request.reset_mock()
    await client.async_get_car_data()
    assert "/config-service/config/flags" in _requested_paths(session)


async def test_get_car_data_propagates_auth_error_from_detection() -> None:
    """Auth failure during detection is not masked by the legacy fallback."""
    session = MagicMock()
    session.request = MagicMock(return_value=_mock_response(401, {}))
    session.post = MagicMock(return_value=_mock_response(401, {}))

    client = _make_client(session)
    with pytest.raises(VoyahApiAuthError):
        await client.async_get_car_data()


# ── Request handling ─────────────────────────────────────────────────────────


async def test_request_refreshes_token_on_401() -> None:
    """On 401, client refreshes token and retries."""
    resp_401 = _mock_response(401, {})
    resp_ok = _mock_response(200, MOCK_CAR_INFO_DATA)
    refresh_resp = _mock_response(200, {"accessToken": "new-access", "refreshToken": "new-refresh"})

    session = MagicMock()
    session.request = MagicMock(side_effect=[resp_401, resp_ok])
    session.post = MagicMock(return_value=refresh_resp)

    client = _make_client(session)
    result = await client.async_get_car_info()

    assert result["imei"] == MOCK_IMEI
    assert client.access_token == "new-access"
    assert client.refresh_token == "new-refresh"


async def test_request_raises_on_unexpected_status() -> None:
    """Non-2xx responses raise VoyahApiError."""
    session = MagicMock()
    session.request = MagicMock(return_value=_mock_response(500, {}))

    client = _make_client(session)
    with pytest.raises(VoyahApiError):
        await client.async_get_car_info()


async def test_get_car_info_success() -> None:
    """async_get_car_info returns dict from API on 200."""
    car_info = {"liveSensors": {"soh": 98}}
    session = MagicMock()
    session.request = MagicMock(return_value=_mock_response(200, car_info))

    client = _make_client(session)
    result = await client.async_get_car_info()
    assert result == {"liveSensors": {"soh": 98}}


# ── Commands ─────────────────────────────────────────────────────────────────


async def test_send_command_uses_telemetry_commands_endpoint() -> None:
    """On the new API, commands are posted by name and IMEI (201 is a success)."""
    session = _routed_session(_new_telemetry_routes())
    client = _make_client(session)

    result = await client.async_send_command("heating", "heatingOn")

    assert result == {"commandId": "cmd-1"}
    command_call = session.request.call_args_list[-1]
    assert command_call.args == ("POST", f"{API_BASE_URL}/client-bff-service/telemetry-commands")
    assert command_call.kwargs["json"] == {"commandName": "heatingOn", "imei": MOCK_IMEI}


async def test_send_command_uses_legacy_tbox_endpoint() -> None:
    """On the legacy API, commands go to the TBox endpoint."""
    session = _routed_session(_legacy_routes())
    client = _make_client(session)

    await client.async_send_command("trunkToggle", "trunkOpen")

    command_call = session.request.call_args_list[-1]
    assert command_call.args == ("POST", f"{API_BASE_URL}/car-service/tbox/{MOCK_CAR_ID}/trunkToggle")


async def test_send_command_without_tbox_command_on_legacy_api() -> None:
    """Commands that exist only in the new API fail clearly on the legacy API."""
    session = _routed_session(_legacy_routes())
    client = _make_client(session)

    with pytest.raises(VoyahApiError):
        await client.async_send_command(None, "search")


async def test_send_tbox_command_posts_to_command_endpoint() -> None:
    """async_send_tbox_command posts to the selected TBox command endpoint."""
    session = MagicMock()
    session.request = MagicMock(return_value=_mock_response(200, {"ok": True}))

    client = _make_client(session)
    result = await client.async_send_tbox_command("cooling")

    assert result == {"ok": True}
    args = session.request.call_args.args
    assert args == ("POST", f"https://app.voyahassist.ru/car-service/tbox/{MOCK_CAR_ID}/cooling")


# ── Helpers ──────────────────────────────────────────────────────────────────


async def test_user_id_from_token() -> None:
    """User id is read from the JWT payload; malformed tokens give None."""
    assert _user_id_from_token(_fake_jwt({"_id": "user-1"})) == "user-1"
    assert _user_id_from_token("not-a-jwt") is None
    assert _user_id_from_token("a.!!!.c") is None
    assert _user_id_from_token(_fake_jwt({"sub": "x"})) is None


# ── Auth helpers ─────────────────────────────────────────────────────────────


async def test_request_sms_raises_connection_error_on_5xx() -> None:
    """async_request_sms raises VoyahApiConnectionError on server error."""
    resp = _mock_response(500, {})
    session = MagicMock()
    session.post = MagicMock(return_value=resp)

    with pytest.raises(VoyahApiConnectionError):
        await VoyahApiClient.async_request_sms(session, "79001234567")


async def test_sign_in_raises_auth_error_on_403() -> None:
    """async_sign_in raises VoyahApiAuthError on 403."""
    resp = _mock_response(403, {"message": "Invalid code"})
    session = MagicMock()
    session.post = MagicMock(return_value=resp)

    with pytest.raises(VoyahApiAuthError):
        await VoyahApiClient.async_sign_in(session, "79001234567", "000000")


async def test_sign_in_returns_tokens_on_success() -> None:
    """async_sign_in returns token dict on 200."""
    tokens = {"accessToken": "acc", "refreshToken": "ref"}
    resp = _mock_response(200, tokens)
    session = MagicMock()
    session.post = MagicMock(return_value=resp)

    result = await VoyahApiClient.async_sign_in(session, "79001234567", "123456")
    assert result["accessToken"] == "acc"
