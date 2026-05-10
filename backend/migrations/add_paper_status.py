"""add status column to papers

Revision ID: add_paper_status
Revises:
Create Date: 2025-01-01 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = 'add_paper_status'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'papers',
        sa.Column('status', sa.String(20), nullable=False, server_default='ready')
    )


def downgrade():
    op.drop_column('papers', 'status')