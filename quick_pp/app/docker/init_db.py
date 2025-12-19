import os

import psycopg2
from dotenv import load_dotenv
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


def create_database():
    load_dotenv()
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    db_name = os.getenv("POSTGRES_DB")
    # Prefer an explicit QPP_DATABASE_URL if provided; otherwise use the Docker
    # postgres service hostname ("postgres") so this script can run inside a
    # container and reach the DB service.
    database_url = os.getenv("QPP_DATABASE_URL")
    if not database_url:
        pg_host = os.getenv("POSTGRES_HOST", "postgres")
        database_url = f"postgresql://{user}:{password}@{pg_host}:5432/{db_name}"

    # Parse the connection string
    if "?" in database_url:
        database_url = database_url.split("?")[0]

    parts = database_url.split("/")
    db_name = parts[-1]
    connection_str = (
        "/".join(parts[:-1]) + "/postgres"
    )  # Connect to default postgres db first

    # Create database if it doesn't exist
    conn = psycopg2.connect(connection_str)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    # Check if database exists
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
    if not cur.fetchone():
        cur.execute(f'CREATE DATABASE "{db_name}"')

    cur.close()
    conn.close()

    # Connect to the actual database
    conn = psycopg2.connect(database_url)
    cur = conn.cursor()

    # Enable pgcrypto extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

    # Create StepType enum
    cur.execute("""
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'steptype') THEN
            CREATE TYPE steptype AS ENUM (
                'assistant_message',
                'embedding',
                'llm',
                'retrieval',
                'rerank',
                'run',
                'system_message',
                'tool',
                'undefined',
                'user_message'
            );
        END IF;
    END $$;
    """)

    # Create tables
    cur.execute("""
    CREATE TABLE IF NOT EXISTS "User" (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        "createdAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        "updatedAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        metadata JSONB NOT NULL,
        identifier TEXT NOT NULL UNIQUE
    );

    CREATE TABLE IF NOT EXISTS "Thread" (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        "createdAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        "updatedAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        "deletedAt" TIMESTAMP WITH TIME ZONE,
        name TEXT,
        metadata JSONB NOT NULL,
        tags TEXT[] DEFAULT ARRAY[]::TEXT[],
        "userId" UUID REFERENCES "User"(id)
    );

    CREATE TABLE IF NOT EXISTS "Step" (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        "createdAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        "updatedAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        "parentId" UUID REFERENCES "Step"(id) ON DELETE CASCADE,
        "threadId" UUID REFERENCES "Thread"(id) ON DELETE CASCADE,
        input TEXT,
        metadata JSONB NOT NULL,
        name TEXT,
        output TEXT,
        type steptype NOT NULL,
        "showInput" TEXT DEFAULT 'json',
        "isError" BOOLEAN DEFAULT FALSE,
        "startTime" TIMESTAMP WITH TIME ZONE NOT NULL,
        "endTime" TIMESTAMP WITH TIME ZONE NOT NULL
    );

    CREATE TABLE IF NOT EXISTS "Element" (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        "createdAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        "updatedAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        "threadId" UUID REFERENCES "Thread"(id) ON DELETE CASCADE,
        "stepId" UUID NOT NULL REFERENCES "Step"(id) ON DELETE CASCADE,
        metadata JSONB NOT NULL,
        mime TEXT,
        name TEXT NOT NULL,
        "objectKey" TEXT,
        url TEXT,
        "chainlitKey" TEXT,
        display TEXT,
        size TEXT,
        language TEXT,
        page INTEGER,
        props JSONB
    );

    CREATE TABLE IF NOT EXISTS "Feedback" (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        "createdAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        "updatedAt" TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        "stepId" UUID REFERENCES "Step"(id),
        name TEXT NOT NULL,
        value FLOAT NOT NULL,
        comment TEXT
    );
    """)

    # Create indexes
    cur.execute("""
    CREATE INDEX IF NOT EXISTS "Element_stepId_idx" ON "Element"("stepId");
    CREATE INDEX IF NOT EXISTS "Element_threadId_idx" ON "Element"("threadId");
    CREATE INDEX IF NOT EXISTS "User_identifier_idx" ON "User"(identifier);
    CREATE INDEX IF NOT EXISTS "Feedback_createdAt_idx" ON "Feedback"("createdAt");
    CREATE INDEX IF NOT EXISTS "Feedback_name_idx" ON "Feedback"(name);
    CREATE INDEX IF NOT EXISTS "Feedback_stepId_idx" ON "Feedback"("stepId");
    CREATE INDEX IF NOT EXISTS "Feedback_value_idx" ON "Feedback"(value);
    CREATE INDEX IF NOT EXISTS "Feedback_name_value_idx" ON "Feedback"(name, value);
    CREATE INDEX IF NOT EXISTS "Step_createdAt_idx" ON "Step"("createdAt");
    CREATE INDEX IF NOT EXISTS "Step_endTime_idx" ON "Step"("endTime");
    CREATE INDEX IF NOT EXISTS "Step_parentId_idx" ON "Step"("parentId");
    CREATE INDEX IF NOT EXISTS "Step_startTime_idx" ON "Step"("startTime");
    CREATE INDEX IF NOT EXISTS "Step_threadId_idx" ON "Step"("threadId");
    CREATE INDEX IF NOT EXISTS "Step_type_idx" ON "Step"(type);
    CREATE INDEX IF NOT EXISTS "Step_name_idx" ON "Step"(name);
    CREATE INDEX IF NOT EXISTS "Step_threadId_startTime_endTime_idx" ON "Step"("threadId", "startTime", "endTime");
    CREATE INDEX IF NOT EXISTS "Thread_createdAt_idx" ON "Thread"("createdAt");
    CREATE INDEX IF NOT EXISTS "Thread_name_idx" ON "Thread"(name);
    """)

    # Connect to default database with autocommit to create new database for langflow
    admin_conn = psycopg2.connect(connection_str)
    admin_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    admin_cur = admin_conn.cursor()
    admin_cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'langflow') THEN
                CREATE ROLE langflow LOGIN PASSWORD 'langflow';
            END IF;
        END
        $$;
    """)
    # Check if database exists first
    admin_cur.execute("SELECT 1 FROM pg_database WHERE datname = 'langflow'")
    if not admin_cur.fetchone():
        # Create database directly without using DO block since CREATE DATABASE can't be in transaction
        admin_cur.execute("CREATE DATABASE langflow OWNER langflow")
    admin_cur.close()
    admin_conn.close()

    conn.commit()
    cur.close()
    conn.close()
    print("Database initialization completed successfully!")


if __name__ == "__main__":
    try:
        create_database()
    except Exception as e:
        print(f"Error initializing database: {e}")
