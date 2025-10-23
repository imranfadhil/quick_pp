-- =============================================================================
-- POSTGRESQL DATABASE SCHEMA FOR QUICK_PP APPLICATION
--
-- This script creates all the necessary tables and relationships
-- for a multi-project, multi-well application.
-- =============================================================================

-- Use a transaction to ensure atomicity
BEGIN;

-- -----------------------------------------------------------------------------
-- Clean Slate: Drop existing tables in reverse order of dependency
-- -----------------------------------------------------------------------------
DROP TRIGGER IF EXISTS "trg_set_projects_timestamp" ON "projects";
DROP TRIGGER IF EXISTS "trg_set_wells_timestamp" ON "wells";
DROP TABLE IF EXISTS "audit_log";
DROP TABLE IF EXISTS "curve_data";
DROP TABLE IF EXISTS "curves";
DROP TABLE IF EXISTS "wells";
DROP TABLE IF EXISTS "project_members";
DROP TABLE IF EXISTS "projects";
DROP TABLE IF EXISTS "users";
DROP FUNCTION IF EXISTS update_updated_at_column();

-- -----------------------------------------------------------------------------
-- TABLE: users
-- Stores user authentication and identity info.
-- -----------------------------------------------------------------------------
CREATE TABLE "users" (
    "user_id" SERIAL PRIMARY KEY,
    "username" VARCHAR(50) UNIQUE NOT NULL,
    "email" VARCHAR(255) UNIQUE NOT NULL,
    "hashed_password" VARCHAR(255) NOT NULL,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------------------------------------
-- TABLE: projects
-- The main container for a collection of wells.
-- -----------------------------------------------------------------------------
CREATE TABLE "projects" (
    "project_id" SERIAL PRIMARY KEY,
    "name" VARCHAR(255) NOT NULL,
    "description" TEXT,
    "created_by" INTEGER REFERENCES "users"("user_id") ON DELETE SET NULL,
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------------------------------------
-- TABLE: project_members (Junction Table)
-- Links users to projects (many-to-many relationship) and defines their role.
-- -----------------------------------------------------------------------------
CREATE TABLE "project_members" (
    "project_id" INTEGER NOT NULL REFERENCES "projects"("project_id") ON DELETE CASCADE,
    "user_id" INTEGER NOT NULL REFERENCES "users"("user_id") ON DELETE CASCADE,
    "role" VARCHAR(50) NOT NULL DEFAULT 'viewer' CHECK (role IN ('admin', 'editor', 'viewer')),
    PRIMARY KEY ("project_id", "user_id")
);

-- -----------------------------------------------------------------------------
-- TABLE: wells
-- The central well object, linked to one project.
-- -----------------------------------------------------------------------------
CREATE TABLE "wells" (
    "well_id" SERIAL PRIMARY KEY,
    "project_id" INTEGER NOT NULL REFERENCES "projects"("project_id") ON DELETE CASCADE,
    "name" VARCHAR(255) NOT NULL,
    "uwi" VARCHAR(100) UNIQUE NOT NULL,
    "header_data" JSONB, -- Flexible JSONB for all messy LAS header info
    "config_data" JSONB, -- Use JSONB to store WellConfig
    "depth_uom" VARCHAR(50),
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------------------------------------
-- TABLE: curves (Metadata)
-- Stores the *metadata* for each curve, linked to one well.
-- The actual data points are in 'curve_data'.
-- -----------------------------------------------------------------------------
CREATE TABLE "curves" (
    "curve_id" SERIAL PRIMARY KEY,
    "well_id" INTEGER NOT NULL REFERENCES "wells"("well_id") ON DELETE CASCADE,
    "mnemonic" VARCHAR(100) NOT NULL,
    "unit" VARCHAR(50),
    "description" TEXT,    
    "data_type" VARCHAR(50) NOT NULL DEFAULT 'numeric' CHECK (data_type IN ('numeric', 'text')),
    UNIQUE("well_id", "mnemonic")
);

-- -----------------------------------------------------------------------------
-- TABLE: curve_data (The "Long" Data Table)
-- Stores the actual data points for every curve.
-- -----------------------------------------------------------------------------
CREATE TABLE "curve_data" (
    "curve_data_id" SERIAL PRIMARY KEY,
    "curve_id" INTEGER NOT NULL REFERENCES "curves"("curve_id") ON DELETE CASCADE,
    "depth" REAL NOT NULL,
    "value_numeric" REAL,
    "value_text" TEXT,
    UNIQUE("curve_id", "depth"),
    -- Ensure that for any row, either the numeric or the text value is populated, but not both.
    CHECK ((value_numeric IS NOT NULL AND value_text IS NULL) OR (value_numeric IS NULL AND value_text IS NOT NULL) OR (value_numeric IS NULL AND value_text IS NULL))
);

-- -----------------------------------------------------------------------------
-- TABLE: audit_log
-- Records all changes.
-- !! IMPORTANT: Your application MUST write to this table manually.
-- !! Triggers were removed as SQLite cannot get the 'user_id'.
-- -----------------------------------------------------------------------------
CREATE TABLE "audit_log" (
    "log_id" SERIAL PRIMARY KEY,
    "user_id" INTEGER REFERENCES "users"("user_id") ON DELETE SET NULL,
    "action" VARCHAR(10) NOT NULL, -- 'INSERT', 'UPDATE', 'DELETE'
    "table_name" VARCHAR(100) NOT NULL,
    "record_id" TEXT,
    "timestamp" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "changes" JSONB -- Stores old and new values: {"old": {...}, "new": {...}}
);

-- -----------------------------------------------------------------------------
-- INDEXES
-- Speed up common queries.
-- -----------------------------------------------------------------------------
CREATE INDEX "idx_wells_project_id" ON "wells" ("project_id");
CREATE INDEX "idx_curves_well_id" ON "curves" ("well_id");
CREATE INDEX "idx_project_members_user_id" ON "project_members" ("user_id");

-- Indexes for the 'curve_data' table
CREATE INDEX "idx_curve_data_curve_id" ON "curve_data" ("curve_id");
CREATE INDEX "idx_curve_data_curve_id_depth" ON "curve_data" ("curve_id", "depth");

-- Indexes for the 'audit_log' table
CREATE INDEX "idx_audit_log_user_table" ON "audit_log" ("user_id", "table_name");
CREATE INDEX "idx_audit_log_record_id" ON "audit_log" ("record_id");

-- =============================================================================
-- AUTOMATION: TRIGGERS
-- =============================================================================

-- -----------------------------------------------------------------------------
-- FUNCTION: update_updated_at_column
-- This function is called by triggers to set the 'updated_at' field.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- -----------------------------------------------------------------------------
-- TRIGGER 1: Auto-update the 'updated_at' timestamp for 'projects'
-- -----------------------------------------------------------------------------
CREATE TRIGGER "trg_set_projects_timestamp"
BEFORE UPDATE ON "projects"
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- -----------------------------------------------------------------------------
-- TRIGGER 2: Auto-update the 'updated_at' timestamp for 'wells'
-- -----------------------------------------------------------------------------
CREATE TRIGGER "trg_set_wells_timestamp"
BEFORE UPDATE ON "wells"
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

COMMIT;
