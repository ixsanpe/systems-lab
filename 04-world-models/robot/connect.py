"""Step 0a: log in to the ECOVACS cloud and find the T90.

The robot is not reached over the LAN: everything (state, commands, live video)
goes through ECOVACS' cloud with your app account. This script only logs in and
lists devices, so it is safe to run.

Usage:
    uv run robot/connect.py

Email and country are asked once and saved to `.secrets/login.json`
(ECOVACS_EMAIL / ECOVACS_COUNTRY env vars override them).

The password is prompted (never put it in a file). On first login ECOVACS may
email a verification code; the resulting account token is cached in
`.secrets/` (gitignored) so later runs skip the code.
"""

import asyncio
import getpass
import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path

import aiohttp
from deebot_client.api_client import ApiClient
from deebot_client.authentication import Authenticator, create_rest_config
from deebot_client.exceptions import DeviceVerificationRequiredError
from deebot_client.models import AccountCredentials
from deebot_client.util import md5

SECRETS = Path(__file__).resolve().parent.parent / ".secrets"
DEVICE_ID_FILE = SECRETS / "device_id"  # stable fake "phone" id, so verification sticks
ACCOUNT_FILE = SECRETS / "account.json"
LOGIN_FILE = SECRETS / "login.json"  # email + country, so you only type them once


def email_and_country() -> tuple[str, str]:
    """Env vars win, then the saved file, then a prompt. Saves whatever was used."""
    saved = json.loads(LOGIN_FILE.read_text()) if LOGIN_FILE.exists() else {}
    email = os.environ.get("ECOVACS_EMAIL") or saved.get("email") or input("ECOVACS account email: ")
    country = (
        os.environ.get("ECOVACS_COUNTRY") or saved.get("country") or input("Country (2 letters, e.g. CH): ")
    ).upper()
    if saved != {"email": email, "country": country}:
        SECRETS.mkdir(mode=0o700, exist_ok=True)
        LOGIN_FILE.write_text(json.dumps({"email": email, "country": country}))
        LOGIN_FILE.chmod(0o600)
    return email, country


def _device_id() -> str:
    SECRETS.mkdir(mode=0o700, exist_ok=True)
    if not DEVICE_ID_FILE.exists():
        DEVICE_ID_FILE.write_text(uuid.uuid4().hex)
    return DEVICE_ID_FILE.read_text().strip()


def _load_account() -> AccountCredentials | None:
    if ACCOUNT_FILE.exists():
        return AccountCredentials(**json.loads(ACCOUNT_FILE.read_text()))
    return None


def _save_account(account: AccountCredentials | None) -> None:
    if account is not None:
        ACCOUNT_FILE.write_text(json.dumps(asdict(account)))
        ACCOUNT_FILE.chmod(0o600)


async def main() -> None:
    email, country = email_and_country()
    password = getpass.getpass("ECOVACS password: ")

    async with aiohttp.ClientSession() as session:
        config = create_rest_config(
            session, device_id=_device_id(), alpha_2_country=country.upper()
        )
        auth = Authenticator(
            config, email, md5(password), account_credentials=_load_account()
        )
        try:
            await auth.authenticate()
        except DeviceVerificationRequiredError:
            await auth.request_device_verification_code()
            code = input("ECOVACS emailed you a verification code, enter it: ")
            await auth.verify_device(code)
        _save_account(auth.account_credentials)
        print("Logged in.\n")

        devices = await ApiClient(auth).get_devices()
        rows = [(d.api, "supported (mqtt)") for d in devices.mqtt]
        rows += [(d, "legacy (xmpp)") for d in devices.xmpp]
        rows += [(d, "NOT in deebot-client") for d in devices.not_supported]
        if not rows:
            print("No devices returned. Is the robot added to this account in the app?")
        for d, status in rows:
            print(
                f"- {d.get('nick') or d.get('deviceName')}: class={d.get('class')} "
                f"did={d.get('did')} resource={d.get('resource')} -> {status}"
            )


if __name__ == "__main__":
    asyncio.run(main())
