"""Associate chunks with sources and normalize interrupted learning states.

Revision ID: 0004_web_ai_shared_sources
Revises: 0003_web_ai_trainer
"""
from alembic import op
import sqlalchemy as sa


revision = "0004_web_ai_shared_sources"
down_revision = "0003_web_ai_trainer"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "web_knowledge_chunks",
        sa.Column("source_ids_json", sa.Text(), nullable=False, server_default="[]"),
    )
    op.execute(
        "UPDATE web_knowledge_chunks "
        "SET source_ids_json = '[' || source_id || ']'"
    )
    op.execute("UPDATE web_sources SET status = 'failed' WHERE status = 'error'")
    op.execute("UPDATE web_sources SET status = 'pending' WHERE status IN ('discovered', 'skipped')")
    op.execute(
        "UPDATE web_sources SET status = 'failed', "
        "error = COALESCE(error, 'A previous learning task was interrupted; retry this source.') "
        "WHERE status = 'running'"
    )
    op.execute(
        "UPDATE web_sources SET status = 'pending' "
        "WHERE kind = 'query' AND status = 'processed'"
    )
    op.execute(
        "UPDATE web_ai_models SET status = 'failed', "
        "status_message = 'A previous learning task was interrupted; retry its sources.' "
        "WHERE status IN ('queued', 'running')"
    )
    with op.batch_alter_table("web_knowledge_chunks") as batch_op:
        batch_op.alter_column("source_ids_json", server_default=None)


def downgrade() -> None:
    op.execute("UPDATE web_sources SET status = 'error' WHERE status = 'failed'")
    op.execute(
        "UPDATE web_sources SET status = 'running' "
        "WHERE status IN ('retrieving', 'retrieved', 'processing')"
    )
    op.drop_column("web_knowledge_chunks", "source_ids_json")
