"""Database schema. The analyses table only ever receives inserts."""

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Analysis(Base):
    __tablename__ = "analyses"
    __table_args__ = (UniqueConstraint("audio_hash", "pipeline_version", name="uq_hash_pipeline"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # SHA-256 of the raw uploaded bytes.
    audio_hash: Mapped[str] = mapped_column(String(64), index=True)
    pipeline_version: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # The descriptor payload returned by the API (no audio, no file name).
    features: Mapped[dict] = mapped_column(JSON)

    @property
    def created_at_utc(self) -> datetime:
        # SQLite drops the timezone; values are always stored in UTC.
        ts = self.created_at
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
