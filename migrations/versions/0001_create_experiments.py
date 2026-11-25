"""Create experiments table.

Revision ID: 0001_create_experiments
Revises:
"""
from alembic import op
import sqlalchemy as sa


revision = "0001_create_experiments"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "experiments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_experiments_name", "experiments", ["name"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_experiments_name", table_name="experiments")
    op.drop_table("experiments")
