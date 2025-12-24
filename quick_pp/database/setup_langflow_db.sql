-- =============================================================================
-- LANGFLOW DATABASE AND USER INITIALIZATION FOR POSTGRESQL
-- This script creates the langflow user, database, and grants privileges.
-- =============================================================================

-- Use a transaction to ensure atomicity
BEGIN;

-- Enable pgcrypto extension (for gen_random_uuid)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pgcrypto') THEN
        EXECUTE 'CREATE EXTENSION pgcrypto';
    END IF;
END
$$;

-- Create StepType enum if not exists
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

-- Create tables
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

-- Create indexes
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

COMMIT;
