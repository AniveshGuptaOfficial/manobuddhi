"""F4: per-path GPU energy (NVML energy counter), for measured savings (O2)

Revision ID: 0002
Revises: 0001
"""
import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("path_telemetry", sa.Column("energy_j", sa.Float))


def downgrade():
    op.drop_column("path_telemetry", "energy_j")
