import os
import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing from environment variables or .env file.")


def get_connection():
    """
    New connection with:
      - dict rows
      - autocommit=True  -> use `with conn.transaction():` for atomic blocks
        (NEVER use `with conn:` in psycopg3: it CLOSES the connection on exit)
      - pgvector registered
    """
    conn = psycopg.connect(
        DATABASE_URL,
        row_factory=dict_row,
        autocommit=True,
        connect_timeout=15,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
        prepare_threshold=None,  # safe with Neon's pooled (pgbouncer) endpoint
    )
    register_vector(conn)
    return conn