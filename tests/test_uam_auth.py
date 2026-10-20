import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from quick_pp.app.backend.main import app
from quick_pp.app.backend.services import auth as auth_service
from quick_pp.app.backend.services import database as db_service
from quick_pp.app.backend.utils import auth as auth_utils
from quick_pp.app.backend.utils.auth import get_totp_code
from quick_pp.database.models import Base
from quick_pp.database.objects import Project


@pytest.fixture
def uam_client(monkeypatch):
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    class DummyCtx:
        def __init__(self, s):
            self._s = s

        def __enter__(self):
            return self._s

        def __exit__(self, exc_type, exc, tb):
            self._s.commit()
            return False

    monkeypatch.setattr(auth_utils, "get_db", lambda: DummyCtx(session))
    monkeypatch.setattr(auth_service, "get_db", lambda: DummyCtx(session))
    monkeypatch.setattr(db_service, "get_db", lambda: DummyCtx(session))

    proj1 = Project(db_session=session, name="Volve_Field")
    proj1.save()
    proj2 = Project(db_session=session, name="Secret_Exploration")
    proj2.save()
    session.commit()

    client = TestClient(app)
    yield client, proj1.project_id, proj2.project_id
    session.close()


def test_cognito_uam_full_lifecycle(uam_client):
    client, pid1, pid2 = uam_client

    # 1. Check bootstrap status creates initial admin in FORCE_CHANGE_PASSWORD state
    r_boot = client.get("/quick_pp/auth/bootstrap-status")
    assert r_boot.status_code == 200
    boot_data = r_boot.json()
    assert boot_data["is_first_run"] is True
    assert boot_data["admin_username"] == "admin"

    # 2. Admin initiates login with temporary bootstrap password -> NEW_PASSWORD_REQUIRED
    r_login = client.post(
        "/quick_pp/auth/login",
        json={"username": "admin", "password": "QuickPP@Admin2026!"},
    )
    assert r_login.status_code == 200
    step1 = r_login.json()
    assert step1["status"] == "CHALLENGE"
    assert step1["challenge_name"] == "NEW_PASSWORD_REQUIRED"
    challenge_token_1 = step1["challenge_token"]

    # 3. Weak password should be rejected by Cognito policy validator
    r_weak = client.post(
        "/quick_pp/auth/challenge/new-password",
        json={"challenge_token": challenge_token_1, "new_password": "weakpassword"},
    )
    assert r_weak.status_code == 400

    # 4. Valid permanent password transitions to MFA_SETUP with QR SVG data URI
    r_pwd = client.post(
        "/quick_pp/auth/challenge/new-password",
        json={
            "challenge_token": challenge_token_1,
            "new_password": "StrongAdmin#2026!",
            "full_name": "Chief Petrophysicist",
        },
    )
    assert r_pwd.status_code == 200
    step2 = r_pwd.json()
    assert step2["status"] == "CHALLENGE"
    assert step2["challenge_name"] == "MFA_SETUP"
    mfa_secret = step2["mfa_setup"]["secret"]
    assert step2["mfa_setup"]["qr_data_uri"].startswith("data:image/svg+xml;base64,")
    assert "otpauth://totp/" in step2["mfa_setup"]["otpauth_uri"]

    # 5. Complete MFA_SETUP using RFC 6238 6-digit TOTP code
    totp_code = get_totp_code(mfa_secret)
    r_mfa_setup = client.post(
        "/quick_pp/auth/challenge/mfa-verify-setup",
        json={
            "challenge_token": step2["challenge_token"],
            "totp_code": totp_code,
        },
    )
    assert r_mfa_setup.status_code == 200
    auth_res = r_mfa_setup.json()
    assert auth_res["status"] == "AUTHENTICATED"
    admin_token = auth_res["access_token"]
    assert auth_res["user"]["status"] == "CONFIRMED"
    assert auth_res["user"]["mfa_enabled"] is True

    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # 6. Subsequent login requires SOFTWARE_TOKEN_MFA
    r_login2 = client.post(
        "/quick_pp/auth/login",
        json={"username": "admin", "password": "StrongAdmin#2026!"},
    )
    assert r_login2.status_code == 200
    assert r_login2.json()["challenge_name"] == "SOFTWARE_TOKEN_MFA"
    r_mfa2 = client.post(
        "/quick_pp/auth/challenge/mfa-verify",
        json={
            "challenge_token": r_login2.json()["challenge_token"],
            "totp_code": get_totp_code(mfa_secret),
        },
    )
    assert r_mfa2.status_code == 200
    assert r_mfa2.json()["status"] == "AUTHENTICATED"

    # 7. Admin invites a new user assigned ONLY to pid1 (Volve_Field) as editor
    r_invite = client.post(
        "/quick_pp/uam/users/invite",
        headers=admin_headers,
        json={
            "username": "geologist1",
            "email": "geo1@company.com",
            "full_name": "Field Geologist",
            "role": "user",
            "project_assignments": [{"project_id": pid1, "role": "editor"}],
        },
    )
    assert r_invite.status_code == 200
    invite_payload = r_invite.json()
    invited_user_id = invite_payload["user"]["user_id"]
    invite_token = invite_payload["invitation"]["invite_token"]
    assert invite_payload["user"]["status"] == "FORCE_CHANGE_PASSWORD"

    # 8. Invited user signs in using direct invite_token link -> NEW_PASSWORD_REQUIRED -> MFA_SETUP -> AUTHENTICATED
    r_u_login = client.post(
        "/quick_pp/auth/login",
        json={"username": "geologist1", "invite_token": invite_token},
    )
    assert r_u_login.status_code == 200
    assert r_u_login.json()["challenge_name"] == "NEW_PASSWORD_REQUIRED"

    r_u_pwd = client.post(
        "/quick_pp/auth/challenge/new-password",
        json={
            "challenge_token": r_u_login.json()["challenge_token"],
            "new_password": "GeoUser#2026!",
        },
    )
    assert r_u_pwd.status_code == 200
    assert r_u_pwd.json()["challenge_name"] == "MFA_SETUP"
    user_mfa_secret = r_u_pwd.json()["mfa_setup"]["secret"]

    r_u_mfa = client.post(
        "/quick_pp/auth/challenge/mfa-verify-setup",
        json={
            "challenge_token": r_u_pwd.json()["challenge_token"],
            "totp_code": get_totp_code(user_mfa_secret),
        },
    )
    assert r_u_mfa.status_code == 200
    user_token = r_u_mfa.json()["access_token"]
    user_headers = {"Authorization": f"Bearer {user_token}"}

    # 9. Standard user only sees assigned project (Volve_Field, pid1), not Secret_Exploration (pid2)
    r_user_projs = client.get("/quick_pp/database/projects", headers=user_headers)
    assert r_user_projs.status_code == 200
    user_projs = r_user_projs.json()
    assert len(user_projs) == 1
    assert user_projs[0]["project_id"] == pid1
    assert user_projs[0]["user_role"] == "editor"

    # Standard user cannot access Admin UAM endpoints
    r_forbidden = client.get("/quick_pp/uam/users", headers=user_headers)
    assert r_forbidden.status_code == 403

    # 10. Admin can reset user MFA, disable user, and view audit logs
    r_reset_mfa = client.post(
        f"/quick_pp/uam/users/{invited_user_id}/reset-mfa", headers=admin_headers
    )
    assert r_reset_mfa.status_code == 200
    assert r_reset_mfa.json()["user"]["mfa_enabled"] is False
    assert r_reset_mfa.json()["user"]["status"] == "MFA_SETUP_REQUIRED"

    r_disable = client.patch(
        f"/quick_pp/uam/users/{invited_user_id}",
        headers=admin_headers,
        json={"is_active": False},
    )
    assert r_disable.status_code == 200
    assert r_disable.json()["user"]["status"] == "DISABLED"

    # Disabled user's existing JWT token is immediately rejected
    r_me_disabled = client.get("/quick_pp/auth/me", headers=user_headers)
    assert r_me_disabled.status_code == 401

    # Audit log has recorded events
    r_audit = client.get("/quick_pp/uam/audit-logs", headers=admin_headers)
    assert r_audit.status_code == 200
    assert len(r_audit.json()["logs"]) >= 5

