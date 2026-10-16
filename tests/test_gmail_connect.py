"""Regression: Gmail connect used connected_accounts.initiate(), which Composio retired (HTTP 400)."""

import json

from server.config import get_settings
from server.models import GmailConnectPayload
from server.services.gmail import client as gmail_client


class FakeConnectedAccounts:
    def __init__(self):
        self.link_args = None

    def initiate(self, **kwargs):
        raise RuntimeError("400 Creating connections on this endpoint ... is no longer supported")

    def link(self, **kwargs):
        self.link_args = kwargs

        class Request:
            redirect_url = "https://connect.example.test/link/lk_1"
            id = "ca_1"

        return Request()


def test_connect_uses_the_supported_link_flow(monkeypatch):
    accounts = FakeConnectedAccounts()
    fake_client = type("C", (), {"connected_accounts": accounts})()
    monkeypatch.setattr(gmail_client, "_get_composio_client", lambda settings=None: fake_client)
    settings = get_settings().model_copy(update={"composio_gmail_auth_config_id": "ac_test"})

    response = gmail_client.initiate_connect(GmailConnectPayload(user_id="web-1"), settings)

    body = json.loads(response.body)
    assert response.status_code == 200 and body["ok"] is True
    assert body["redirect_url"] == "https://connect.example.test/link/lk_1"
    assert accounts.link_args["auth_config_id"] == "ac_test" and accounts.link_args["user_id"] == "web-1"
    assert gmail_client.get_active_gmail_user_id() == "web-1"


def test_connect_without_an_auth_config_is_a_clean_400(monkeypatch):
    settings = get_settings().model_copy(update={"composio_gmail_auth_config_id": None})
    response = gmail_client.initiate_connect(GmailConnectPayload(), settings)
    assert response.status_code == 400
