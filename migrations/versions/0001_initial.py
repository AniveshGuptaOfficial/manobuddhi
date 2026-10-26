"""F1 initial schema: eval_runs, requests, path_telemetry, settings_history

Revision ID: 0001
Revises:
"""
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

TS = sa.DateTime(timezone=True)


def upgrade():
    op.create_table(
        "eval_runs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("benchmark", sa.String(64), nullable=False, index=True),
        sa.Column("split", sa.String(32), nullable=False),
        sa.Column("n_items", sa.Integer, nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("progress", sa.Float, nullable=False),
        sa.Column("router_config", sa.JSON, nullable=False),
        sa.Column("baseline", sa.String(32)),
        sa.Column("summary", sa.JSON),
        sa.Column("git_commit", sa.String(40)),
        sa.Column("created_at", TS, server_default=sa.func.now(), nullable=False),
        sa.Column("finished_at", TS),
    )
    op.create_table(
        "requests",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("created_at", TS, server_default=sa.func.now(), nullable=False, index=True),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("eval_run_id", sa.Integer, sa.ForeignKey("eval_runs.id", ondelete="CASCADE"),
                  index=True),
        sa.Column("item_id", sa.String(128)),
        sa.Column("query", sa.Text, nullable=False),
        sa.Column("mode", sa.String(32), nullable=False),
        sa.Column("fast_model", sa.String(128), nullable=False),
        sa.Column("slow_model", sa.String(128), nullable=False),
        sa.Column("route", sa.String(16), nullable=False),
        sa.Column("conflict_score", sa.Float),
        sa.Column("probe_score", sa.Float),
        sa.Column("disagreement", sa.Float),
        sa.Column("k_samples", sa.Integer),
        sa.Column("logprob_confidence", sa.Float),
        sa.Column("threshold", sa.Float),
        sa.Column("threshold_mode", sa.String(16)),
        sa.Column("latency_budget_ms", sa.Float),
        sa.Column("reuse_method", sa.String(32)),
        sa.Column("fast_answer", sa.Text),
        sa.Column("final_answer", sa.Text),
        sa.Column("gold", sa.Text),
        sa.Column("correct", sa.Boolean),
        sa.Column("fast_correct", sa.Boolean),
        sa.Column("lure_hit", sa.Boolean),
        sa.Column("total_tokens", sa.Integer),
        sa.Column("total_latency_ms", sa.Float),
        sa.Column("total_gpu_seconds", sa.Float),
    )
    op.create_table(
        "path_telemetry",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("request_id", sa.Integer, sa.ForeignKey("requests.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("path", sa.String(16), nullable=False),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("prompt_tokens", sa.Integer),
        sa.Column("completion_tokens", sa.Integer),
        sa.Column("reused_tokens", sa.Integer),
        sa.Column("ttft_ms", sa.Float),
        sa.Column("latency_ms", sa.Float),
        sa.Column("gpu_seconds", sa.Float),
        sa.Column("gpu_util_pct", sa.Float),
        sa.Column("gpu_mem_mb", sa.Float),
        sa.Column("queue_depth", sa.Integer),
    )
    op.create_table(
        "settings_history",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("created_at", TS, server_default=sa.func.now(), nullable=False),
        sa.Column("settings", sa.JSON, nullable=False),
    )


def downgrade():
    op.drop_table("settings_history")
    op.drop_table("path_telemetry")
    op.drop_table("requests")
    op.drop_table("eval_runs")
