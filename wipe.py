from __future__ import annotations

from database.db import get_connection

# List of all application tables in your schema
TABLES_TO_TRUNCATE = [
    "chunks",
    "requirement_dependencies",
    "traceability",
    "user_stories",
    "action_items",
    "backlog_items",
    "meetings",
    "sprints",
    "revision_history",
    "tech_stack",
    "role_permissions",
    "roles",
    "transitions",
    "states",
    "decisions",
    "documents",
    "projects",
]

def wipe_database():
    conn = get_connection()
    try:
        print("Connecting to Neon Postgres...")
        with conn:
            with conn.cursor() as cur:
                # Truncate all tables at once with CASCADE to handle foreign keys cleanly
                tables_sql = ", ".join(TABLES_TO_TRUNCATE)
                print(f"Truncating tables: {tables_sql}...")
                cur.execute(f"TRUNCATE TABLE {tables_sql} RESTART IDENTITY CASCADE;")
        
        print("✅ Success! All database records and vector data have been completely wiped.")
    except Exception as e:
        print(f"❌ Error wiping database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    print("⚠️  WARNING: This will permanently delete ALL data, documents, and embeddings from your database.")
    confirm = input("Are you sure you want to proceed? Type 'yes' to confirm: ").strip().lower()
    
    if confirm == "yes":
        wipe_database()
    else:
        print("Operation cancelled. No data was deleted.")