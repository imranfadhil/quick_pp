-- =============================================================================
-- SQLITE DATABASE SCHEMA FOR QUICK_PP APPLICATION
--
-- This script creates all the necessary tables and relationships
-- for a multi-project, multi-well application.
-- =============================================================================

BEGIN;

-- -----------------------------------------------------------------------------
-- Clean Slate: Drop existing tables in reverse order of dependency
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS "audit_log";
DROP TABLE IF EXISTS "curve_data";
DROP TABLE IF EXISTS "capillary_pressure";
DROP TABLE IF EXISTS "relative_permeability";
DROP TABLE IF EXISTS "core_measurements";
DROP TABLE IF EXISTS "core_samples";
DROP TABLE IF EXISTS "pressure_tests";
DROP TABLE IF EXISTS "fluid_contacts";
DROP TABLE IF EXISTS "formation_tops";
DROP TABLE IF EXISTS "curves";
DROP TABLE IF EXISTS "wells";
DROP TABLE IF EXISTS "project_members";
DROP TABLE IF EXISTS "projects";
DROP TABLE IF EXISTS "users";

-- -----------------------------------------------------------------------------
-- TABLE: users
-- Stores user authentication and identity info.
-- -----------------------------------------------------------------------------
CREATE TABLE "users" (
    "user_id" INTEGER PRIMARY KEY AUTOINCREMENT,
    "username" TEXT UNIQUE NOT NULL,
    "email" TEXT UNIQUE NOT NULL,
    "hashed_password" TEXT NOT NULL,
    "created_at" DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------------------------------------
-- TABLE: projects
-- The main container for a collection of wells.
-- -----------------------------------------------------------------------------
CREATE TABLE "projects" (
    "project_id" INTEGER PRIMARY KEY AUTOINCREMENT,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "created_by" INTEGER REFERENCES "users"("user_id") ON DELETE SET NULL,
    "created_at" DATETIME DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------------------------------------
-- TABLE: project_members (Junction Table)
-- Links users to projects (many-to-many relationship) and defines their role.
-- -----------------------------------------------------------------------------
CREATE TABLE "project_members" (
    "project_id" INTEGER NOT NULL REFERENCES "projects"("project_id") ON DELETE CASCADE,
    "user_id" INTEGER NOT NULL REFERENCES "users"("user_id") ON DELETE CASCADE,
    "role" TEXT NOT NULL DEFAULT 'viewer' CHECK (role IN ('admin', 'editor', 'viewer')),
    PRIMARY KEY ("project_id", "user_id")
);

-- -----------------------------------------------------------------------------
-- TABLE: wells
-- The central well object, linked to one project.
-- -----------------------------------------------------------------------------
CREATE TABLE "wells" (
    "well_id" INTEGER PRIMARY KEY AUTOINCREMENT,
    "project_id" INTEGER NOT NULL REFERENCES "projects"("project_id") ON DELETE CASCADE,
    "name" TEXT NOT NULL,
    "uwi" TEXT UNIQUE NOT NULL,
    "header_data" JSON, -- Flexible JSON for all messy LAS header info
    "config_data" JSON, -- Use TEXT to store WellConfig as JSON
    "depth_uom" TEXT,
    "created_at" DATETIME DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------------------------------------
-- TABLE: curves (Metadata)
-- Stores the *metadata* for each curve, linked to one well.
-- The actual data points are in 'curve_data'.
-- -----------------------------------------------------------------------------
CREATE TABLE "curves" (
    "curve_id" INTEGER PRIMARY KEY AUTOINCREMENT,
    "well_id" INTEGER NOT NULL REFERENCES "wells"("well_id") ON DELETE CASCADE,
    "mnemonic" TEXT NOT NULL,
    "unit" TEXT,
    "description" TEXT,    
    "data_type" TEXT NOT NULL DEFAULT 'numeric' CHECK (data_type IN ('numeric', 'text')),
    UNIQUE("well_id", "mnemonic")
);

-- -----------------------------------------------------------------------------
-- TABLE: curve_data (The "Long" Data Table)
-- Stores the actual data points for every curve.
-- -----------------------------------------------------------------------------
CREATE TABLE "curve_data" (
    "curve_data_id" INTEGER PRIMARY KEY AUTOINCREMENT,
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
CREATE TABLE "formation_tops" (
    "top_id" INTEGER PRIMARY KEY AUTOINCREMENT,
    "well_id" INTEGER NOT NULL REFERENCES "wells"("well_id") ON DELETE CASCADE,
    "name" TEXT NOT NULL,
    "depth" REAL NOT NULL,
    UNIQUE("well_id", "name")
);

-- -----------------------------------------------------------------------------
-- TABLE: fluid_contacts
-- Stores discrete depth markers for fluid contacts (e.g., OWC, GOC).
-- -----------------------------------------------------------------------------
CREATE TABLE "fluid_contacts" (
    "contact_id" INTEGER PRIMARY KEY AUTOINCREMENT,
    "well_id" INTEGER NOT NULL REFERENCES "wells"("well_id") ON DELETE CASCADE,
    "name" TEXT NOT NULL,
    "depth" REAL NOT NULL,
    UNIQUE("well_id", "name")
);

-- -----------------------------------------------------------------------------
-- TABLE: pressure_tests
-- Stores point pressure measurements.
-- -----------------------------------------------------------------------------
CREATE TABLE "pressure_tests" (
    "test_id" INTEGER PRIMARY KEY AUTOINCREMENT,
    "well_id" INTEGER NOT NULL REFERENCES "wells"("well_id") ON DELETE CASCADE,
    "depth" REAL NOT NULL,
    "pressure" REAL NOT NULL,
    "pressure_uom" TEXT DEFAULT 'psi',
    UNIQUE("well_id", "depth")
);

-- -----------------------------------------------------------------------------
-- TABLE: core_samples
-- Represents a physical core plug taken from a well.
-- -----------------------------------------------------------------------------
CREATE TABLE "core_samples" (
    "sample_id" INTEGER PRIMARY KEY AUTOINCREMENT,
    "well_id" INTEGER NOT NULL REFERENCES "wells"("well_id") ON DELETE CASCADE,
    "sample_name" TEXT NOT NULL,
    "depth" REAL NOT NULL,
    "description" TEXT,
    "remark" TEXT,
    UNIQUE("well_id", "sample_name"),
    UNIQUE("well_id", "depth")
);

-- -----------------------------------------------------------------------------
-- TABLE: core_measurements
-- Stores single-value measurements for a core sample (e.g., porosity).
-- -----------------------------------------------------------------------------
CREATE TABLE "core_measurements" (
    "measurement_id" INTEGER PRIMARY KEY AUTOINCREMENT,
    "sample_id" INTEGER NOT NULL REFERENCES "core_samples"("sample_id") ON DELETE CASCADE,
    "property_name" TEXT NOT NULL,
    "value" REAL NOT NULL,
    "unit" TEXT,
    UNIQUE("sample_id", "property_name")
);

-- -----------------------------------------------------------------------------
-- TABLE: relative_permeability
-- Stores relative permeability data series for a core sample.
-- -----------------------------------------------------------------------------
CREATE TABLE "relative_permeability" (
    "relperm_id" INTEGER PRIMARY KEY AUTOINCREMENT,
    "sample_id" INTEGER NOT NULL REFERENCES "core_samples"("sample_id") ON DELETE CASCADE,
    "saturation" REAL NOT NULL,
    "kr" REAL NOT NULL,
    "phase" TEXT NOT NULL,
    UNIQUE("sample_id", "phase", "saturation")
);

-- -----------------------------------------------------------------------------
-- TABLE: capillary_pressure
-- Stores capillary pressure data series for a core sample.
-- -----------------------------------------------------------------------------
CREATE TABLE "capillary_pressure" (
    "pc_id" INTEGER PRIMARY KEY AUTOINCREMENT,
    "sample_id" INTEGER NOT NULL REFERENCES "core_samples"("sample_id") ON DELETE CASCADE,
    "saturation" REAL NOT NULL,
    "pressure" REAL NOT NULL,
    "experiment_type" TEXT,
    "cycle" TEXT NOT NULL DEFAULT 'drainage' CHECK (cycle IN ('drainage', 'imbibition')),
    UNIQUE("sample_id", "saturation", "experiment_type", "cycle")
);

-- -----------------------------------------------------------------------------
-- TABLE: audit_log
-- Records all changes.
-- !! IMPORTANT: Your application MUST write to this table manually.
-- !! Triggers were removed as SQLite cannot get the 'user_id'.
-- -----------------------------------------------------------------------------
CREATE TABLE "audit_log" (
    "log_id" INTEGER PRIMARY KEY AUTOINCREMENT,
    "user_id" INTEGER REFERENCES "users"("user_id") ON DELETE SET NULL,
    "action" TEXT NOT NULL, -- 'INSERT', 'UPDATE', 'DELETE'
    "table_name" TEXT NOT NULL,
    "record_id" TEXT,
    "timestamp" DATETIME DEFAULT CURRENT_TIMESTAMP,
    "changes" JSON -- Stores old and new values: {"old": {...}, "new": {...}}
);

-- -----------------------------------------------------------------------------
-- INDEXES
-- Speed up common queries.
-- -----------------------------------------------------------------------------
CREATE INDEX "idx_wells_project_id" ON "wells" ("project_id");
CREATE INDEX "idx_curves_well_id" ON "curves" ("well_id");
CREATE INDEX "idx_project_members_user_id" ON "project_members" ("user_id");
CREATE INDEX "idx_formation_tops_well_id" ON "formation_tops" ("well_id");
CREATE INDEX "idx_fluid_contacts_well_id" ON "fluid_contacts" ("well_id");
CREATE INDEX "idx_pressure_tests_well_id" ON "pressure_tests" ("well_id");
CREATE INDEX "idx_core_samples_well_id" ON "core_samples" ("well_id");
CREATE INDEX "idx_core_measurements_sample_id" ON "core_measurements" ("sample_id");
CREATE INDEX "idx_relative_permeability_sample_id" ON "relative_permeability" ("sample_id");
CREATE INDEX "idx_capillary_pressure_sample_id" ON "capillary_pressure" ("sample_id");

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
-- TRIGGER 1: Auto-update the 'updated_at' timestamp for 'projects'
-- -----------------------------------------------------------------------------
CREATE TRIGGER "trg_set_projects_timestamp"
AFTER UPDATE ON "projects"
FOR EACH ROW
BEGIN
    UPDATE "projects"
    SET "updated_at" = CURRENT_TIMESTAMP
    WHERE "project_id" = OLD."project_id";
END;

-- -----------------------------------------------------------------------------
-- TRIGGER 2: Auto-update the 'updated_at' timestamp for 'wells'
-- -----------------------------------------------------------------------------
CREATE TRIGGER "trg_set_wells_timestamp"
AFTER UPDATE ON "wells"
FOR EACH ROW
BEGIN
    UPDATE "wells"
    SET "updated_at" = CURRENT_TIMESTAMP
    WHERE "well_id" = OLD."well_id";
END;

COMMIT;
