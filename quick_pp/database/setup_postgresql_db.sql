-- =============================================================================
-- POSTGRESQL DATABASE SCHEMA FOR QUICK_PP APPLICATION
--
-- This script creates all the necessary tables and relationships
-- for a multi-project, multi-well application.
-- =============================================================================

-- Use a transaction to ensure atomicity
BEGIN;

-- -----------------------------------------------------------------------------
-- TABLE: users
-- Stores user authentication and identity info.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS "users" (
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
CREATE TABLE IF NOT EXISTS "projects" (
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
CREATE TABLE IF NOT EXISTS "project_members" (
    "project_id" INTEGER NOT NULL REFERENCES "projects"("project_id") ON DELETE CASCADE,
    "user_id" INTEGER NOT NULL REFERENCES "users"("user_id") ON DELETE CASCADE,
    "role" VARCHAR(50) NOT NULL DEFAULT 'viewer' CHECK (role IN ('admin', 'editor', 'viewer')),
    PRIMARY KEY ("project_id", "user_id")
);

-- -----------------------------------------------------------------------------
-- TABLE: wells
-- The central well object, linked to one project.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS "wells" (
    "well_id" SERIAL PRIMARY KEY,
    "project_id" INTEGER NOT NULL REFERENCES "projects"("project_id") ON DELETE CASCADE,
    "name" VARCHAR(255) NOT NULL,
    "uwi" VARCHAR(100) NOT NULL,
    "header_data" JSONB, -- Flexible JSONB for all messy LAS header info
    "config_data" JSONB, -- Use JSONB to store WellConfig
    "depth_uom" VARCHAR(50),
    "created_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE("project_id", "uwi")
);

-- -----------------------------------------------------------------------------
-- TABLE: curves (Metadata)
-- Stores the *metadata* for each curve, linked to one well.
-- The actual data points are in 'curve_data'.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS "curves" (
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
CREATE TABLE IF NOT EXISTS "curve_data" (
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
-- TABLE: formation_tops
-- Stores discrete depth markers for formation tops.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS "formation_tops" (
    "top_id" SERIAL PRIMARY KEY,
    "well_id" INTEGER NOT NULL REFERENCES "wells"("well_id") ON DELETE CASCADE,
    "name" VARCHAR(255) NOT NULL,
    "depth" REAL NOT NULL,
    UNIQUE("well_id", "name")
);

-- -----------------------------------------------------------------------------
-- TABLE: fluid_contacts
-- Stores discrete depth markers for fluid contacts (e.g., OWC, GOC).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS "fluid_contacts" (
    "contact_id" SERIAL PRIMARY KEY,
    "well_id" INTEGER NOT NULL REFERENCES "wells"("well_id") ON DELETE CASCADE,
    "name" VARCHAR(100) NOT NULL,
    "depth" REAL NOT NULL,
    UNIQUE("well_id", "name")
);

-- -----------------------------------------------------------------------------
-- TABLE: pressure_tests
-- Stores point pressure measurements.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS "pressure_tests" (
    "test_id" SERIAL PRIMARY KEY,
    "well_id" INTEGER NOT NULL REFERENCES "wells"("well_id") ON DELETE CASCADE,
    "depth" REAL NOT NULL,
    "pressure" REAL NOT NULL,
    "pressure_uom" VARCHAR(50) DEFAULT 'psi',
    UNIQUE("well_id", "depth")
);

-- -----------------------------------------------------------------------------
-- TABLE: well_surveys
-- Stores directional survey points (MD, Inc, Azim).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS "well_surveys" (
    "survey_id" SERIAL PRIMARY KEY,
    "well_id" INTEGER NOT NULL REFERENCES "wells"("well_id") ON DELETE CASCADE,
    "md" REAL NOT NULL, -- Measured Depth
    "inc" REAL NOT NULL, -- Inclination
    "azim" REAL NOT NULL, -- Azimuth
    UNIQUE("well_id", "md")
);


-- -----------------------------------------------------------------------------
-- TABLE: core_samples
-- Represents a physical core plug taken from a well.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS "core_samples" (
    "sample_id" SERIAL PRIMARY KEY,
    "well_id" INTEGER NOT NULL REFERENCES "wells"("well_id") ON DELETE CASCADE,
    "sample_name" VARCHAR(100) NOT NULL,
    "depth" REAL NOT NULL,
    "description" TEXT,
    "remark" TEXT,
    UNIQUE("well_id", "sample_name", "depth")
);

-- -----------------------------------------------------------------------------
-- TABLE: core_measurements
-- Stores single-value measurements for a core sample (e.g., porosity).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS "core_measurements" (
    "measurement_id" SERIAL PRIMARY KEY,
    "sample_id" INTEGER NOT NULL REFERENCES "core_samples"("sample_id") ON DELETE CASCADE,
    "property_name" VARCHAR(100) NOT NULL,
    "value" REAL NOT NULL,
    "unit" VARCHAR(50),
    UNIQUE("sample_id", "property_name")
);

-- -----------------------------------------------------------------------------
-- TABLE: relative_permeability
-- Stores relative permeability data series for a core sample.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS "relative_permeability" (
    "relperm_id" SERIAL PRIMARY KEY,
    "sample_id" INTEGER NOT NULL REFERENCES "core_samples"("sample_id") ON DELETE CASCADE,
    "saturation" REAL NOT NULL,
    "kr" REAL NOT NULL,
    "phase" VARCHAR(50) NOT NULL,
    UNIQUE("sample_id", "phase", "saturation")
);

-- -----------------------------------------------------------------------------
-- TABLE: capillary_pressure
-- Stores capillary pressure data series for a core sample.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS "capillary_pressure" (
    "pc_id" SERIAL PRIMARY KEY,
    "sample_id" INTEGER NOT NULL REFERENCES "core_samples"("sample_id") ON DELETE CASCADE,
    "saturation" REAL NOT NULL,
    "pressure" REAL NOT NULL,
    "experiment_type" VARCHAR(50),
    "cycle" VARCHAR(50) NOT NULL DEFAULT 'drainage' CHECK (cycle IN ('drainage', 'imbibition')),
    UNIQUE("sample_id", "saturation", "experiment_type", "cycle")
);

-- -----------------------------------------------------------------------------
-- TABLE: audit_log
-- Records all changes.
-- !! IMPORTANT: Your application MUST write to this table manually.
-- !! Triggers were removed as SQLite cannot get the 'user_id'.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS "audit_log" (
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
CREATE INDEX IF NOT EXISTS "idx_wells_project_id" ON "wells" ("project_id");
CREATE INDEX IF NOT EXISTS "idx_curves_well_id" ON "curves" ("well_id");
CREATE INDEX IF NOT EXISTS "idx_project_members_user_id" ON "project_members" ("user_id");
CREATE INDEX IF NOT EXISTS "idx_formation_tops_well_id" ON "formation_tops" ("well_id");
CREATE INDEX IF NOT EXISTS "idx_fluid_contacts_well_id" ON "fluid_contacts" ("well_id");
CREATE INDEX IF NOT EXISTS "idx_pressure_tests_well_id" ON "pressure_tests" ("well_id");
CREATE INDEX IF NOT EXISTS "idx_well_surveys_well_id" ON "well_surveys" ("well_id");
CREATE INDEX IF NOT EXISTS "idx_core_samples_well_id" ON "core_samples" ("well_id");
CREATE INDEX IF NOT EXISTS "idx_core_measurements_sample_id" ON "core_measurements" ("sample_id");
CREATE INDEX IF NOT EXISTS "idx_relative_permeability_sample_id" ON "relative_permeability" ("sample_id");
CREATE INDEX IF NOT EXISTS "idx_capillary_pressure_sample_id" ON "capillary_pressure" ("sample_id");

-- Indexes for the 'curve_data' table
CREATE INDEX IF NOT EXISTS "idx_curve_data_curve_id" ON "curve_data" ("curve_id");
CREATE INDEX IF NOT EXISTS "idx_curve_data_curve_id_depth" ON "curve_data" ("curve_id", "depth");

-- Indexes for the 'audit_log' table
CREATE INDEX IF NOT EXISTS "idx_audit_log_user_table" ON "audit_log" ("user_id", "table_name");
CREATE INDEX IF NOT EXISTS "idx_audit_log_record_id" ON "audit_log" ("record_id");

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
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'trg_set_projects_timestamp'
    ) THEN
        EXECUTE 'CREATE TRIGGER trg_set_projects_timestamp
            BEFORE UPDATE ON projects
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();';
    END IF;
END
$$;

-- -----------------------------------------------------------------------------
-- TRIGGER 2: Auto-update the 'updated_at' timestamp for 'wells'
-- -----------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'trg_set_wells_timestamp'
    ) THEN
        EXECUTE 'CREATE TRIGGER trg_set_wells_timestamp
            BEFORE UPDATE ON wells
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();';
    END IF;
END
$$;

COMMIT;
