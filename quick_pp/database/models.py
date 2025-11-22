from datetime import datetime
from typing import List, Optional, Dict, Any

from sqlalchemy import (
    Integer,
    String,
    JSON,
    REAL,
    Text,
    ForeignKey,
    UniqueConstraint,
    CheckConstraint,
)
from sqlalchemy.orm import declarative_base, relationship, Mapped, mapped_column

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        Text, default=datetime.now, nullable=False
    )

    projects: Mapped[List["Project"]] = relationship(
        "Project", back_populates="creator"
    )
    project_memberships: Mapped[List["ProjectMember"]] = relationship(
        "ProjectMember", back_populates="user"
    )
    audit_logs: Mapped[List["AuditLog"]] = relationship(
        "AuditLog", back_populates="user"
    )

    def __repr__(self):
        return f"<User(user_id={self.user_id}, username='{self.username}')>"


class Project(Base):
    __tablename__ = "projects"

    project_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    created_by: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.user_id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(
        Text, default=datetime.now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        Text, default=datetime.now, onupdate=datetime.now, nullable=False
    )

    creator: Mapped[Optional["User"]] = relationship("User", back_populates="projects")
    members: Mapped[List["ProjectMember"]] = relationship(
        "ProjectMember", back_populates="project"
    )
    wells: Mapped[List["Well"]] = relationship(
        "Well", back_populates="project", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Project(project_id={self.project_id}, name='{self.name}')>"


class ProjectMember(Base):
    __tablename__ = "project_members"

    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.project_id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True
    )
    role: Mapped[str] = mapped_column(String(50), nullable=False, default="viewer")

    __table_args__ = (
        CheckConstraint(role.in_(["admin", "editor", "viewer"]), name="role_check"),
    )

    project: Mapped["Project"] = relationship("Project", back_populates="members")
    user: Mapped["User"] = relationship("User", back_populates="project_memberships")

    def __repr__(self):
        return f"<ProjectMember(project_id={self.project_id}, user_id={self.user_id}, role='{self.role}')>"


class Well(Base):
    __tablename__ = "wells"

    well_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.project_id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    uwi: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    depth_uom: Mapped[Optional[str]] = mapped_column(String(50))

    header_data: Mapped[Optional[Dict[str, Any]]] = mapped_column("header_data", JSON)
    config_data: Mapped[Optional[Dict[str, Any]]] = mapped_column("config_data", JSON)

    created_at: Mapped[datetime] = mapped_column(
        Text, default=datetime.now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        Text, default=datetime.now, onupdate=datetime.now
    )

    project: Mapped["Project"] = relationship("Project", back_populates="wells")
    curves: Mapped[List["Curve"]] = relationship(
        "Curve", back_populates="well", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Well(well_id={self.well_id}, name='{self.name}', uwi='{self.uwi}')>"


class Curve(Base):
    __tablename__ = "curves"

    curve_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    well_id: Mapped[int] = mapped_column(
        ForeignKey("wells.well_id", ondelete="CASCADE"), nullable=False
    )
    mnemonic: Mapped[str] = mapped_column(String(100), nullable=False)
    unit: Mapped[Optional[str]] = mapped_column(String(50))
    description: Mapped[Optional[str]] = mapped_column(Text)
    data_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default="numeric"
    )

    __table_args__ = (
        UniqueConstraint("well_id", "mnemonic", name="uq_well_id_mnemonic"),
        CheckConstraint(data_type.in_(["numeric", "text"]), name="data_type_check"),
    )

    well: Mapped["Well"] = relationship("Well", back_populates="curves")
    data: Mapped[List["CurveData"]] = relationship(
        "CurveData", back_populates="curve", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Curve(curve_id={self.curve_id}, well_id={self.well_id}, mnemonic='{self.mnemonic}')>"


class CurveData(Base):
    __tablename__ = "curve_data"

    curve_data_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    curve_id: Mapped[int] = mapped_column(
        ForeignKey("curves.curve_id", ondelete="CASCADE"), nullable=False
    )
    depth: Mapped[float] = mapped_column(REAL, nullable=False)
    value_numeric: Mapped[Optional[float]] = mapped_column(REAL)
    value_text: Mapped[Optional[str]] = mapped_column(Text)

    __table_args__ = (
        UniqueConstraint("curve_id", "depth", name="uq_curve_id_depth"),
        CheckConstraint(
            "(value_numeric IS NOT NULL AND value_text IS NULL) OR "
            "(value_numeric IS NULL AND value_text IS NOT NULL) OR "
            "(value_numeric IS NULL AND value_text IS NULL)",
            name="check_one_value_is_null",
        ),
    )

    curve: Mapped["Curve"] = relationship("Curve", back_populates="data")

    def __repr__(self):
        return (
            f"<CurveData(curve_id={self.curve_id}, depth={self.depth}, "
            f"num={self.value_numeric}, txt='{self.value_text}')>"
        )


class AuditLog(Base):
    __tablename__ = "audit_log"

    log_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.user_id", ondelete="SET NULL")
    )
    action: Mapped[str] = mapped_column(String(10), nullable=False)
    table_name: Mapped[str] = mapped_column(String(100), nullable=False)
    record_id: Mapped[Optional[str]] = mapped_column(Text)
    timestamp: Mapped[datetime] = mapped_column(
        Text, default=datetime.now, nullable=False
    )
    changes: Mapped[Optional[Dict[str, Any]]] = mapped_column("changes", JSON)

    user: Mapped[Optional["User"]] = relationship("User", back_populates="audit_logs")

    def __repr__(self):
        return f"<AuditLog(log_id={self.log_id}, table='{self.table_name}', action='{self.action}')>"
