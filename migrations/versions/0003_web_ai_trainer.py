"""Add independent Web AI Trainer storage.

Revision ID: 0003_web_ai_trainer
Revises: 0002_neural_lab_resources
"""
from alembic import op
import sqlalchemy as sa


revision = "0003_web_ai_trainer"
down_revision = "0002_neural_lab_resources"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "web_ai_models",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("topic", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("status_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "web_sources",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("parent_id", sa.Integer(), nullable=True),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("selected", sa.Boolean(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["model_id"], ["web_ai_models.id"]),
        sa.ForeignKeyConstraint(["parent_id"], ["web_sources.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_web_sources_model_id", "web_sources", ["model_id"])
    op.create_table(
        "web_knowledge_chunks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["model_id"], ["web_ai_models.id"]),
        sa.ForeignKeyConstraint(["source_id"], ["web_sources.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_web_knowledge_chunks_model_id", "web_knowledge_chunks", ["model_id"])
    op.create_index("ix_web_knowledge_chunks_source_id", "web_knowledge_chunks", ["source_id"])


def downgrade() -> None:
    op.drop_index("ix_web_knowledge_chunks_source_id", table_name="web_knowledge_chunks")
    op.drop_index("ix_web_knowledge_chunks_model_id", table_name="web_knowledge_chunks")
    op.drop_table("web_knowledge_chunks")
    op.drop_index("ix_web_sources_model_id", table_name="web_sources")
    op.drop_table("web_sources")
    op.drop_table("web_ai_models")
