"""Engine setup and cache operations. Only lookups and inserts: nothing is updated or deleted."""

from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Analysis, Base


def make_sessionmaker(database_url: str) -> sessionmaker[Session]:
    kwargs = {"connect_args": {"check_same_thread": False}} if database_url.startswith("sqlite") else {}
    engine: Engine = create_engine(database_url, **kwargs)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


def lookup(session: Session, audio_hash: str, pipeline_version: str) -> Analysis | None:
    stmt = (
        select(Analysis)
        .where(Analysis.audio_hash == audio_hash, Analysis.pipeline_version == pipeline_version)
        .order_by(Analysis.created_at.desc())
        .limit(1)
    )
    return session.scalars(stmt).first()


def insert(session: Session, audio_hash: str, pipeline_version: str, features: dict) -> tuple[Analysis, bool]:
    """Insert a result. Returns (row, inserted). If a concurrent request stored the same
    (hash, pipeline version) first, that row is returned instead and nothing is written."""
    row = Analysis(audio_hash=audio_hash, pipeline_version=pipeline_version, features=features)
    session.add(row)
    try:
        session.commit()
        return row, True
    except IntegrityError:
        session.rollback()
        existing = lookup(session, audio_hash, pipeline_version)
        if existing is None:
            raise
        return existing, False
