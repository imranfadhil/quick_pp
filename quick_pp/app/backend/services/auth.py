"""AWS Cognito-style User Access Management (UAM) & Authentication API Router.

Implements:
- Invite-only authentication with `NEW_PASSWORD_REQUIRED`, `MFA_SETUP`, and `SOFTWARE_TOKEN_MFA` challenges
- TOTP Authenticator App provisioning with inline SVG QR codes
- Admin User Management (Invite user, regenerate invite/temp password, reset MFA, enable/disable, role assignment)
- Project-level Role-Based Access Control (`admin`, `editor`, `viewer`)
- Security Audit Log viewing
"""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import desc, func, select

from quick_pp.app.backend.utils.auth import (
    AUTH_COOKIE_NAME,
    bootstrap_default_admin,
    build_otpauth_uri,
    create_jwt_token,
    decode_jwt_token,
    generate_temp_password,
    generate_totp_secret,
    get_current_user,
    hash_password,
    record_audit_event,
    require_admin,
    send_invitation_email,
    serialize_user,
    validate_password_policy,
    verify_password,
    verify_totp_code,
)
from quick_pp.app.backend.utils.db import get_db
from quick_pp.app.backend.utils.qr_svg import generate_qr_data_uri
from quick_pp.database import models as db_models

router = APIRouter(tags=["Authentication & UAM"])


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    username: str = Field(..., description="Username or email address")
    password: Optional[str] = Field(None, description="User password or temporary password")
    invite_token: Optional[str] = Field(
        None, description="Optional one-time invite token from invitation URL"
    )


class NewPasswordChallengeRequest(BaseModel):
    challenge_token: str = Field(..., description="Challenge token from NEW_PASSWORD_REQUIRED")
    new_password: str = Field(..., description="New permanent password")
    full_name: Optional[str] = Field(None, description="Optional user full name")


class MfaVerifyChallengeRequest(BaseModel):
    challenge_token: str = Field(..., description="Challenge token from MFA_SETUP or SOFTWARE_TOKEN_MFA")
    totp_code: str = Field(..., description="6-digit code from Authenticator app")


class ChangeOwnPasswordRequest(BaseModel):
    current_password: str
    new_password: str
    totp_code: str = Field(..., description="6-digit MFA code to confirm password change")


class ProjectRoleAssignment(BaseModel):
    project_id: int
    role: str = Field("viewer", description="'viewer', 'editor', or 'admin'")


class AdminInviteUserRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)
    email: str = Field(..., min_length=3, max_length=255)
    full_name: Optional[str] = Field("")
    role: str = Field("user", description="Global role: 'user' or 'admin'")
    project_assignments: List[ProjectRoleAssignment] = Field(default_factory=list)
    custom_temp_password: Optional[str] = Field(None)
    base_url: Optional[str] = Field(None, description="Frontend origin URL for building invite link")


class AdminUpdateUserRequest(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None
    project_assignments: Optional[List[ProjectRoleAssignment]] = None


class ProjectMemberUpsertRequest(BaseModel):
    user_id: int
    role: str = Field("viewer", description="'viewer', 'editor', or 'admin'")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_auth_cookie(response: Response, access_token: str) -> None:
    """Attach HttpOnly session cookie to response."""
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=access_token,
        httponly=True,
        max_age=43200,
        samesite="lax",
        path="/",
    )


def _build_mfa_setup_payload(session, user: db_models.User) -> Dict[str, Any]:
    """Ensure `user` has a TOTP secret and return the MFA_SETUP challenge payload."""
    if not user.mfa_secret:
        user.mfa_secret = generate_totp_secret()
    user.status = "MFA_SETUP_REQUIRED"
    user.mfa_enabled = 0
    session.add(user)
    session.flush()

    otpauth_uri = build_otpauth_uri(username=user.username, secret=user.mfa_secret)
    qr_data_uri = generate_qr_data_uri(otpauth_uri)
    challenge_token = create_jwt_token(
        subject=user.user_id,
        token_type="challenge_mfa_setup",
        expires_in_seconds=900,
        extra_claims={"username": user.username},
    )
    return {
        "status": "CHALLENGE",
        "challenge_name": "MFA_SETUP",
        "challenge_token": challenge_token,
        "message": "Scan the QR code with your Authenticator app and enter the 6-digit verification code.",
        "mfa_setup": {
            "secret": user.mfa_secret,
            "otpauth_uri": otpauth_uri,
            "qr_data_uri": qr_data_uri,
            "issuer": "quick_pp",
            "account_name": user.username,
        },
        "user": serialize_user(user, session=session),
    }


def _build_invite_url(request: Request, base_url: Optional[str], username: str, invite_token: str) -> str:
    origin = (
        (base_url or "").strip().rstrip("/")
        or os.environ.get("PUBLIC_URL", "").strip().rstrip("/")
        or str(request.base_url).rstrip("/")
    )
    return f"{origin}/login?username={quote(username)}&invite={quote(invite_token)}"


# ---------------------------------------------------------------------------
# Public Authentication & Challenge Endpoints
# ---------------------------------------------------------------------------


@router.get("/auth/bootstrap-status", summary="Check initial bootstrap admin status")
def auth_bootstrap_status():
    """Ensure default admin is bootstrapped if DB is empty and report whether first-time setup is pending."""
    bootstrap_default_admin()
    with get_db() as session:
        total_users = session.scalar(select(func.count(db_models.User.user_id))) or 0
        admin_user = session.scalar(
            select(db_models.User).where(db_models.User.role == "admin").limit(1)
        )
        is_first_run = bool(
            total_users == 1
            and admin_user is not None
            and admin_user.status == "FORCE_CHANGE_PASSWORD"
            and not admin_user.last_login_at
        )
        return {
            "is_first_run": is_first_run,
            "admin_username": admin_user.username if is_first_run and admin_user else None,
            "default_password_hint": (
                "QuickPP@Admin2026! (or QPP_ADMIN_PASSWORD in .env)"
                if is_first_run
                else None
            ),
        }


@router.post("/auth/login", summary="Initiate Cognito-style authentication")
def auth_login(req: LoginRequest):
    """Step 1 of Cognito-style login: verify password (or invite token) and issue next challenge."""
    bootstrap_default_admin()
    ident = req.username.strip()
    if not ident:
        raise HTTPException(status_code=400, detail="Username or email is required")

    with get_db() as session:
        user = session.scalar(
            select(db_models.User).where(
                (func.lower(db_models.User.username) == ident.lower())
                | (func.lower(db_models.User.email) == ident.lower())
            )
        )
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password. Access is by administrator invitation only.",
            )

        if not user.is_active or user.status == "DISABLED":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This account has been disabled by an administrator.",
            )

        # Authenticate via password OR valid invite_token (when in FORCE_CHANGE_PASSWORD state)
        authenticated_via_invite = False
        if (
            req.invite_token
            and user.invite_token
            and user.status == "FORCE_CHANGE_PASSWORD"
            and secrets.compare_digest(req.invite_token.strip(), user.invite_token)
        ):
            authenticated_via_invite = True
        elif req.password and verify_password(req.password, user.hashed_password):
            authenticated_via_invite = False
        else:
            record_audit_event(
                session,
                user_id=user.user_id,
                action="LOGIN_FAIL",
                table_name="users",
                record_id=str(user.user_id),
                changes={"reason": "invalid_credentials"},
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password.",
            )

        # Challenge 1: Temporary password / first-time invite -> NEW_PASSWORD_REQUIRED
        if user.status == "FORCE_CHANGE_PASSWORD" or authenticated_via_invite:
            challenge_token = create_jwt_token(
                subject=user.user_id,
                token_type="challenge_new_password",
                expires_in_seconds=900,
                extra_claims={"username": user.username},
            )
            return {
                "status": "CHALLENGE",
                "challenge_name": "NEW_PASSWORD_REQUIRED",
                "challenge_token": challenge_token,
                "message": "Welcome! Please set a new permanent password for your account.",
                "user": serialize_user(user, session=session),
            }

        # Challenge 2: MFA not yet configured -> MFA_SETUP
        if user.status == "MFA_SETUP_REQUIRED" or not user.mfa_enabled or not user.mfa_secret:
            return _build_mfa_setup_payload(session, user)

        # Challenge 3: Standard MFA verification -> SOFTWARE_TOKEN_MFA
        challenge_token = create_jwt_token(
            subject=user.user_id,
            token_type="challenge_mfa_verify",
            expires_in_seconds=600,
            extra_claims={"username": user.username},
        )
        return {
            "status": "CHALLENGE",
            "challenge_name": "SOFTWARE_TOKEN_MFA",
            "challenge_token": challenge_token,
            "message": "Enter the 6-digit verification code from your Authenticator app.",
            "user": serialize_user(user, session=session),
        }


@router.post(
    "/auth/challenge/new-password",
    summary="Complete NEW_PASSWORD_REQUIRED challenge and transition to MFA setup",
)
def auth_challenge_new_password(req: NewPasswordChallengeRequest):
    """Set permanent password for an invited user and transition to MFA challenge."""
    try:
        payload = decode_jwt_token(
            req.challenge_token, expected_type="challenge_new_password"
        )
        user_id = int(payload["sub"])
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)
        ) from exc

    ok, policy_errors = validate_password_policy(req.new_password)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=" ".join(policy_errors),
        )

    with get_db() as session:
        user = session.get(db_models.User, user_id)
        if not user or not user.is_active or user.status == "DISABLED":
            raise HTTPException(status_code=403, detail="User account is unavailable.")

        if verify_password(req.new_password, user.hashed_password):
            raise HTTPException(
                status_code=400,
                detail="New permanent password must be different from your temporary password.",
            )

        user.hashed_password = hash_password(req.new_password)
        user.invite_token = None
        if req.full_name is not None and req.full_name.strip():
            user.full_name = req.full_name.strip()

        record_audit_event(
            session,
            user_id=user.user_id,
            action="PWD_SET",
            table_name="users",
            record_id=str(user.user_id),
            changes={"status": "password_updated_from_invite"},
        )

        # If MFA is already enabled (e.g. admin re-invited user without resetting MFA), prompt for 6-digit code
        if user.mfa_enabled and user.mfa_secret:
            user.status = "CONFIRMED"
            session.add(user)
            session.flush()
            challenge_token = create_jwt_token(
                subject=user.user_id,
                token_type="challenge_mfa_verify",
                expires_in_seconds=600,
                extra_claims={"username": user.username},
            )
            return {
                "status": "CHALLENGE",
                "challenge_name": "SOFTWARE_TOKEN_MFA",
                "challenge_token": challenge_token,
                "message": "Password updated. Enter the 6-digit code from your Authenticator app.",
                "user": serialize_user(user, session=session),
            }

        # Otherwise transition directly to mandatory TOTP MFA enrollment
        return _build_mfa_setup_payload(session, user)


@router.post(
    "/auth/challenge/mfa-verify-setup",
    summary="Verify TOTP code to complete MFA_SETUP enrollment and issue access token",
)
def auth_challenge_mfa_verify_setup(req: MfaVerifyChallengeRequest, response: Response):
    """Confirm newly enrolled TOTP Authenticator code and complete user sign-in."""
    try:
        payload = decode_jwt_token(
            req.challenge_token, expected_type="challenge_mfa_setup"
        )
        user_id = int(payload["sub"])
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)
        ) from exc

    with get_db() as session:
        user = session.get(db_models.User, user_id)
        if not user or not user.is_active or user.status == "DISABLED":
            raise HTTPException(status_code=403, detail="User account is unavailable.")

        if not user.mfa_secret or not verify_totp_code(user.mfa_secret, req.totp_code):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid 6-digit verification code. Check your Authenticator app and try again.",
            )

        now_iso = datetime.now(timezone.utc).isoformat()
        user.mfa_enabled = 1
        user.status = "CONFIRMED"
        user.invite_token = None
        user.last_login_at = now_iso
        session.add(user)
        session.flush()

        record_audit_event(
            session,
            user_id=user.user_id,
            action="MFA_ENROLL",
            table_name="users",
            record_id=str(user.user_id),
            changes={"mfa_enabled": True, "status": "CONFIRMED"},
        )

        access_token = create_jwt_token(
            subject=user.user_id,
            token_type="access",
            expires_in_seconds=43200,
            extra_claims={"username": user.username, "role": user.role},
        )
        _set_auth_cookie(response, access_token)
        return {
            "status": "AUTHENTICATED",
            "access_token": access_token,
            "user": serialize_user(user, session=session),
        }


@router.post(
    "/auth/challenge/mfa-verify",
    summary="Verify TOTP code for SOFTWARE_TOKEN_MFA challenge and issue access token",
)
def auth_challenge_mfa_verify(req: MfaVerifyChallengeRequest, response: Response):
    """Verify 6-digit TOTP code on subsequent logins and issue session access token."""
    try:
        payload = decode_jwt_token(
            req.challenge_token, expected_type="challenge_mfa_verify"
        )
        user_id = int(payload["sub"])
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)
        ) from exc

    with get_db() as session:
        user = session.get(db_models.User, user_id)
        if not user or not user.is_active or user.status == "DISABLED":
            raise HTTPException(status_code=403, detail="User account is unavailable.")

        if not user.mfa_secret or not verify_totp_code(user.mfa_secret, req.totp_code):
            record_audit_event(
                session,
                user_id=user.user_id,
                action="MFA_FAIL",
                table_name="users",
                record_id=str(user.user_id),
                changes={"reason": "invalid_totp_code"},
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid 6-digit Authenticator code.",
            )

        user.last_login_at = datetime.now(timezone.utc).isoformat()
        session.add(user)
        session.flush()

        record_audit_event(
            session,
            user_id=user.user_id,
            action="LOGIN",
            table_name="users",
            record_id=str(user.user_id),
            changes={"method": "totp_mfa"},
        )

        access_token = create_jwt_token(
            subject=user.user_id,
            token_type="access",
            expires_in_seconds=43200,
            extra_claims={"username": user.username, "role": user.role},
        )
        _set_auth_cookie(response, access_token)
        return {
            "status": "AUTHENTICATED",
            "access_token": access_token,
            "user": serialize_user(user, session=session),
        }


@router.get("/auth/me", summary="Get current authenticated user profile")
def auth_me(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Return the authenticated user's profile and project memberships."""
    return {"user": current_user}


@router.post("/auth/logout", summary="Sign out and clear session cookie")
def auth_logout(request: Request, response: Response):
    """Clear authentication cookie."""
    response.delete_cookie(key=AUTH_COOKIE_NAME, path="/")
    return {"status": "LOGGED_OUT"}


@router.post("/auth/change-password", summary="Change own password with MFA verification")
def auth_change_own_password(
    req: ChangeOwnPasswordRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Allow an authenticated user to change their password after verifying current password + TOTP code."""
    ok, policy_errors = validate_password_policy(req.new_password)
    if not ok:
        raise HTTPException(status_code=400, detail=" ".join(policy_errors))

    with get_db() as session:
        user = session.get(db_models.User, int(current_user["user_id"]))
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if not verify_password(req.current_password, user.hashed_password):
            raise HTTPException(status_code=400, detail="Current password is incorrect.")

        if not user.mfa_secret or not verify_totp_code(user.mfa_secret, req.totp_code):
            raise HTTPException(status_code=400, detail="Invalid 6-digit MFA code.")

        user.hashed_password = hash_password(req.new_password)
        session.add(user)
        record_audit_event(
            session,
            user_id=user.user_id,
            action="PWD_CHANGE",
            table_name="users",
            record_id=str(user.user_id),
            changes={"by": "self"},
        )
        return {"message": "Password updated successfully."}


# ---------------------------------------------------------------------------
# Admin User Access Management (UAM) Endpoints
# ---------------------------------------------------------------------------


@router.get("/uam/users", summary="List all users and project assignments (Admin only)")
def uam_list_users(admin: Dict[str, Any] = Depends(require_admin)):
    """Return all users in the system with status, MFA state, and project memberships."""
    with get_db() as session:
        users = (
            session.execute(select(db_models.User).order_by(db_models.User.user_id))
            .scalars()
            .all()
        )
        return {"users": [serialize_user(u, session=session) for u in users]}


@router.post("/uam/users/invite", summary="Invite a new user with temporary password & invite link (Admin only)")
def uam_invite_user(
    req: AdminInviteUserRequest,
    request: Request,
    admin: Dict[str, Any] = Depends(require_admin),
):
    """Create a new invited user in `FORCE_CHANGE_PASSWORD` status and return credentials + invite link."""
    username = req.username.strip()
    email = req.email.strip().lower()
    role = req.role.strip().lower()
    if role not in ("user", "admin"):
        raise HTTPException(status_code=400, detail="Role must be 'user' or 'admin'")

    temp_password = (
        req.custom_temp_password.strip()
        if req.custom_temp_password and req.custom_temp_password.strip()
        else generate_temp_password()
    )
    ok, policy_errors = validate_password_policy(temp_password)
    if not ok:
        raise HTTPException(
            status_code=400,
            detail="Temporary password does not meet policy: " + " ".join(policy_errors),
        )

    with get_db() as session:
        existing = session.scalar(
            select(db_models.User).where(
                (func.lower(db_models.User.username) == username.lower())
                | (func.lower(db_models.User.email) == email.lower())
            )
        )
        if existing:
            raise HTTPException(
                status_code=409,
                detail="A user with that username or email already exists.",
            )

        invite_token = secrets.token_urlsafe(24)
        new_user = db_models.User(
            username=username,
            email=email,
            full_name=(req.full_name or "").strip(),
            hashed_password=hash_password(temp_password),
            role=role,
            status="FORCE_CHANGE_PASSWORD",
            is_active=1,
            mfa_enabled=0,
            mfa_secret=None,
            invite_token=invite_token,
            invited_by=int(admin["user_id"]),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        session.add(new_user)
        session.flush()

        # Assign initial project memberships
        for pa in req.project_assignments:
            p_role = pa.role if pa.role in ("viewer", "editor", "admin") else "viewer"
            proj = session.get(db_models.Project, pa.project_id)
            if proj:
                session.add(
                    db_models.ProjectMember(
                        project_id=pa.project_id,
                        user_id=new_user.user_id,
                        role=p_role,
                    )
                )
        session.flush()

        invite_url = _build_invite_url(request, req.base_url, username, invite_token)
        email_sent, email_msg = send_invitation_email(
            to_email=email,
            username=username,
            temp_password=temp_password,
            invite_url=invite_url,
            invited_by_name=admin.get("username", "Administrator"),
        )

        record_audit_event(
            session,
            user_id=int(admin["user_id"]),
            action="INVITE",
            table_name="users",
            record_id=str(new_user.user_id),
            changes={
                "invited_username": username,
                "email": email,
                "role": role,
                "projects": [pa.model_dump() for pa in req.project_assignments],
            },
        )

        return {
            "user": serialize_user(new_user, session=session),
            "invitation": {
                "username": username,
                "email": email,
                "temp_password": temp_password,
                "invite_token": invite_token,
                "invite_url": invite_url,
                "email_sent": email_sent,
                "email_message": email_msg,
            },
        }


@router.post(
    "/uam/users/{user_id}/reinvite",
    summary="Regenerate temporary password and invite link for a user (Admin only)",
)
def uam_reinvite_user(
    user_id: int,
    request: Request,
    base_url: Optional[str] = Query(None),
    admin: Dict[str, Any] = Depends(require_admin),
):
    """Reset a user's password to a new temporary password and generate a fresh invite link."""
    with get_db() as session:
        user = session.get(db_models.User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        temp_password = generate_temp_password()
        invite_token = secrets.token_urlsafe(24)
        user.hashed_password = hash_password(temp_password)
        user.invite_token = invite_token
        user.status = "FORCE_CHANGE_PASSWORD"
        user.is_active = 1
        session.add(user)
        session.flush()

        invite_url = _build_invite_url(request, base_url, user.username, invite_token)
        email_sent, email_msg = send_invitation_email(
            to_email=user.email,
            username=user.username,
            temp_password=temp_password,
            invite_url=invite_url,
            invited_by_name=admin.get("username", "Administrator"),
        )

        record_audit_event(
            session,
            user_id=int(admin["user_id"]),
            action="REINVITE",
            table_name="users",
            record_id=str(user.user_id),
            changes={"username": user.username},
        )

        return {
            "user": serialize_user(user, session=session),
            "invitation": {
                "username": user.username,
                "email": user.email,
                "temp_password": temp_password,
                "invite_token": invite_token,
                "invite_url": invite_url,
                "email_sent": email_sent,
                "email_message": email_msg,
            },
        }


@router.post(
    "/uam/users/{user_id}/reset-mfa",
    summary="Reset a user's TOTP MFA enrollment so they set up a new Authenticator on next login (Admin only)",
)
def uam_reset_user_mfa(
    user_id: int,
    admin: Dict[str, Any] = Depends(require_admin),
):
    """Clear a user's MFA secret and require MFA re-enrollment on their next sign-in."""
    with get_db() as session:
        user = session.get(db_models.User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        user.mfa_secret = None
        user.mfa_enabled = 0
        if user.status == "CONFIRMED":
            user.status = "MFA_SETUP_REQUIRED"
        session.add(user)
        session.flush()

        record_audit_event(
            session,
            user_id=int(admin["user_id"]),
            action="RESET_MFA",
            table_name="users",
            record_id=str(user.user_id),
            changes={"username": user.username},
        )
        return {"user": serialize_user(user, session=session)}


@router.patch(
    "/uam/users/{user_id}",
    summary="Update user role, active status, profile, or project memberships (Admin only)",
)
def uam_update_user(
    user_id: int,
    req: AdminUpdateUserRequest,
    admin: Dict[str, Any] = Depends(require_admin),
):
    """Update a user's global role, active status, or project assignments."""
    with get_db() as session:
        user = session.get(db_models.User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        is_self = int(admin["user_id"]) == user.user_id
        changes: Dict[str, Any] = {}

        if req.full_name is not None:
            user.full_name = req.full_name.strip()
            changes["full_name"] = user.full_name

        if req.email is not None and req.email.strip():
            user.email = req.email.strip().lower()
            changes["email"] = user.email

        if req.role is not None:
            new_role = req.role.strip().lower()
            if new_role not in ("user", "admin"):
                raise HTTPException(status_code=400, detail="Role must be 'user' or 'admin'")
            if is_self and new_role != "admin":
                raise HTTPException(
                    status_code=400, detail="You cannot demote your own administrator account."
                )
            user.role = new_role
            changes["role"] = new_role

        if req.is_active is not None:
            if is_self and not req.is_active:
                raise HTTPException(
                    status_code=400, detail="You cannot disable your own administrator account."
                )
            user.is_active = 1 if req.is_active else 0
            if not req.is_active:
                user.status = "DISABLED"
            elif user.status == "DISABLED":
                user.status = "CONFIRMED" if user.mfa_enabled else "MFA_SETUP_REQUIRED"
            changes["is_active"] = req.is_active
            changes["status"] = user.status

        if req.project_assignments is not None:
            # Replace user's project memberships
            existing_pms = (
                session.execute(
                    select(db_models.ProjectMember).where(
                        db_models.ProjectMember.user_id == user.user_id
                    )
                )
                .scalars()
                .all()
            )
            for pm in existing_pms:
                session.delete(pm)
            session.flush()

            for pa in req.project_assignments:
                p_role = pa.role if pa.role in ("viewer", "editor", "admin") else "viewer"
                if session.get(db_models.Project, pa.project_id):
                    session.add(
                        db_models.ProjectMember(
                            project_id=pa.project_id,
                            user_id=user.user_id,
                            role=p_role,
                        )
                    )
            changes["project_assignments"] = [pa.model_dump() for pa in req.project_assignments]

        session.add(user)
        session.flush()

        record_audit_event(
            session,
            user_id=int(admin["user_id"]),
            action="UPDATE_USR",
            table_name="users",
            record_id=str(user.user_id),
            changes=changes,
        )
        return {"user": serialize_user(user, session=session)}


@router.delete(
    "/uam/users/{user_id}",
    summary="Delete a user account (Admin only)",
)
def uam_delete_user(
    user_id: int,
    admin: Dict[str, Any] = Depends(require_admin),
):
    """Permanently delete a user account."""
    if int(admin["user_id"]) == user_id:
        raise HTTPException(
            status_code=400, detail="You cannot delete your own administrator account."
        )
    with get_db() as session:
        user = session.get(db_models.User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        uname = user.username
        session.delete(user)
        session.flush()
        record_audit_event(
            session,
            user_id=int(admin["user_id"]),
            action="DELETE_USR",
            table_name="users",
            record_id=str(user_id),
            changes={"deleted_username": uname},
        )
        return {"message": f"User '{uname}' deleted."}


@router.get(
    "/uam/projects/{project_id}/members",
    summary="List members of a project (Admin only)",
)
def uam_list_project_members(
    project_id: int,
    admin: Dict[str, Any] = Depends(require_admin),
):
    """Return all members assigned to a specific project."""
    with get_db() as session:
        rows = session.execute(
            select(db_models.ProjectMember, db_models.User)
            .join(
                db_models.User,
                db_models.ProjectMember.user_id == db_models.User.user_id,
            )
            .where(db_models.ProjectMember.project_id == project_id)
            .order_by(db_models.User.username)
        ).all()
        return {
            "project_id": project_id,
            "members": [
                {
                    "user_id": u.user_id,
                    "username": u.username,
                    "email": u.email,
                    "full_name": u.full_name or "",
                    "global_role": u.role,
                    "project_role": pm.role,
                }
                for pm, u in rows
            ],
        }


@router.post(
    "/uam/projects/{project_id}/members",
    summary="Add or update a member's role on a project (Admin only)",
)
def uam_upsert_project_member(
    project_id: int,
    req: ProjectMemberUpsertRequest,
    admin: Dict[str, Any] = Depends(require_admin),
):
    """Assign or update a user's project role ('viewer', 'editor', 'admin')."""
    if req.role not in ("viewer", "editor", "admin"):
        raise HTTPException(
            status_code=400, detail="Project role must be 'viewer', 'editor', or 'admin'"
        )
    with get_db() as session:
        proj = session.get(db_models.Project, project_id)
        if not proj:
            raise HTTPException(status_code=404, detail="Project not found")
        user = session.get(db_models.User, req.user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        pm = session.get(db_models.ProjectMember, (project_id, req.user_id))
        if pm:
            pm.role = req.role
        else:
            pm = db_models.ProjectMember(
                project_id=project_id, user_id=req.user_id, role=req.role
            )
            session.add(pm)
        session.flush()
        record_audit_event(
            session,
            user_id=int(admin["user_id"]),
            action="PROJ_MEMB",
            table_name="project_members",
            record_id=f"{project_id}:{req.user_id}",
            changes={"username": user.username, "project": proj.name, "role": req.role},
        )
        return {"message": "Project member updated", "project_id": project_id, "user_id": req.user_id, "role": req.role}


@router.delete(
    "/uam/projects/{project_id}/members/{user_id}",
    summary="Remove a user from a project (Admin only)",
)
def uam_remove_project_member(
    project_id: int,
    user_id: int,
    admin: Dict[str, Any] = Depends(require_admin),
):
    """Revoke a user's access to a project."""
    with get_db() as session:
        pm = session.get(db_models.ProjectMember, (project_id, user_id))
        if pm:
            session.delete(pm)
            session.flush()
        return {"message": "Member removed from project"}


@router.get("/uam/audit-logs", summary="Get recent security and UAM audit events (Admin only)")
def uam_list_audit_logs(
    limit: int = Query(50, ge=1, le=200),
    admin: Dict[str, Any] = Depends(require_admin),
):
    """Return recent security and UAM audit events."""
    with get_db() as session:
        rows = session.execute(
            select(db_models.AuditLog, db_models.User.username)
            .outerjoin(
                db_models.User,
                db_models.AuditLog.user_id == db_models.User.user_id,
            )
            .order_by(desc(db_models.AuditLog.log_id))
            .limit(limit)
        ).all()
        return {
            "logs": [
                {
                    "log_id": log_row.log_id,
                    "user_id": log_row.user_id,
                    "username": uname or "system",
                    "action": log_row.action,
                    "table_name": log_row.table_name,
                    "record_id": log_row.record_id,
                    "timestamp": str(log_row.timestamp or ""),
                    "changes": log_row.changes or {},
                }
                for log_row, uname in rows
            ]
        }

