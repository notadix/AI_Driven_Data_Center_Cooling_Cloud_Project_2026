-- =============================================================================
-- PostgreSQL Schema — AI-Driven Cooling Digital Twin
-- Project: AI-Driven Cloud-Native Hybrid Cooling Digital Twin for Sustainable Data Centers
-- Author : Snigda Chandanala (Phase 2)
-- =============================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- 1. Facilities
-- =============================================================================
CREATE TABLE IF NOT EXISTS facilities (
    facility_id       VARCHAR(64)   PRIMARY KEY,
    facility_name     VARCHAR(255)  NOT NULL,
    aws_region        VARCHAR(32)   NOT NULL DEFAULT 'us-east-1',
    nominal_capacity_mw NUMERIC(10,4) NOT NULL DEFAULT 5.0,
    target_pue        NUMERIC(5,4)  NOT NULL DEFAULT 1.20,
    latitude          NUMERIC(9,6),
    longitude         NUMERIC(9,6),
    created_at        TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- 2. CRAC Units
-- =============================================================================
CREATE TABLE IF NOT EXISTS crac_units (
    facility_id         VARCHAR(64)  NOT NULL REFERENCES facilities(facility_id) ON DELETE CASCADE,
    crac_id             VARCHAR(64)  NOT NULL,
    cooling_zone        VARCHAR(32)  NOT NULL DEFAULT 'north',
    cooling_capacity_kw NUMERIC(8,2) NOT NULL DEFAULT 120.0,
    rated_flow_lpm      NUMERIC(8,2) NOT NULL DEFAULT 7500.0,
    chiller_cop_rated   NUMERIC(5,3) NOT NULL DEFAULT 3.5,
    sitewise_asset_id   VARCHAR(128),
    twinmaker_entity_id VARCHAR(128),
    is_active           BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (facility_id, crac_id)
);

CREATE INDEX IF NOT EXISTS idx_crac_facility ON crac_units(facility_id);

-- =============================================================================
-- 3. Server Racks
-- =============================================================================
CREATE TABLE IF NOT EXISTS racks (
    facility_id           VARCHAR(64)  NOT NULL REFERENCES facilities(facility_id) ON DELETE CASCADE,
    rack_id               VARCHAR(64)  NOT NULL,
    crac_id               VARCHAR(64)  NOT NULL,
    grid_row              SMALLINT     NOT NULL CHECK (grid_row BETWEEN 0 AND 7),
    grid_col              SMALLINT     NOT NULL CHECK (grid_col BETWEEN 0 AND 7),
    max_thermal_rating_kw NUMERIC(6,2) NOT NULL DEFAULT 20.0,
    world_pos_x           NUMERIC(7,3),
    world_pos_y           NUMERIC(7,3),
    world_pos_z           NUMERIC(7,3),
    created_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (facility_id, rack_id),
    FOREIGN KEY (facility_id, crac_id) REFERENCES crac_units(facility_id, crac_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_rack_crac     ON racks(facility_id, crac_id);
CREATE INDEX IF NOT EXISTS idx_rack_facility ON racks(facility_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_rack_grid ON racks(facility_id, grid_row, grid_col);

-- =============================================================================
-- 4. RL Agent Episodes
-- =============================================================================
CREATE TABLE IF NOT EXISTS rl_episodes (
    episode_id     UUID         PRIMARY KEY DEFAULT uuid_generate_v4(),
    facility_id    VARCHAR(64)  NOT NULL REFERENCES facilities(facility_id) ON DELETE CASCADE,
    model_version  VARCHAR(64)  NOT NULL DEFAULT 'v1.0',
    start_time     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    end_time       TIMESTAMPTZ,
    total_steps    INTEGER      NOT NULL DEFAULT 0,
    mean_pue       NUMERIC(6,4),
    mean_reward    NUMERIC(10,6),
    total_cost     NUMERIC(10,6),
    ashrae_violations INTEGER   NOT NULL DEFAULT 0,
    critical_violations INTEGER NOT NULL DEFAULT 0,
    policy_source  VARCHAR(32)  NOT NULL DEFAULT 'SafePPO',
    metadata       JSONB        NOT NULL DEFAULT '{}'::jsonb,
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_episodes_facility ON rl_episodes(facility_id);
CREATE INDEX IF NOT EXISTS idx_episodes_time     ON rl_episodes(start_time DESC);

-- =============================================================================
-- 5. RL Agent Step Log (per-step action/reward/observation recording)
-- =============================================================================
CREATE TABLE IF NOT EXISTS rl_steps (
    step_id       BIGSERIAL    PRIMARY KEY,
    episode_id    UUID         NOT NULL REFERENCES rl_episodes(episode_id) ON DELETE CASCADE,
    step_index    INTEGER      NOT NULL,
    -- Observations (ASHRAE-relevant)
    server_inlet_temp_c  NUMERIC(6,3),
    fws_supply_temp_c    NUMERIC(6,3),
    return_temp_c        NUMERIC(6,3),
    flow_rate_lpm        NUMERIC(8,2),
    it_power_mw          NUMERIC(10,6),
    -- Actions taken by agent
    action_delta_supply_c  NUMERIC(6,3),
    action_pump_speed_pct  NUMERIC(5,2),
    action_fan_speed_pct   NUMERIC(5,2),
    action_valve_split_pct NUMERIC(5,2),
    -- Reward components
    reward_pue_penalty     NUMERIC(10,6),
    reward_safety_bonus    NUMERIC(10,6),
    reward_total           NUMERIC(10,6),
    -- Safety signals
    lagrangian_cost        NUMERIC(10,6),
    ashrae_violated        BOOLEAN NOT NULL DEFAULT FALSE,
    critical_violated      BOOLEAN NOT NULL DEFAULT FALSE,
    pue                    NUMERIC(6,4),
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_steps_episode  ON rl_steps(episode_id);
CREATE INDEX IF NOT EXISTS idx_steps_violated ON rl_steps(episode_id, ashrae_violated);

-- =============================================================================
-- 6. FNO Surrogate Model Predictions
-- =============================================================================
CREATE TABLE IF NOT EXISTS fno_predictions (
    prediction_id   UUID          PRIMARY KEY DEFAULT uuid_generate_v4(),
    facility_id     VARCHAR(64)   NOT NULL REFERENCES facilities(facility_id) ON DELETE CASCADE,
    crac_id         VARCHAR(64)   NOT NULL,
    predicted_at    TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    model_version   VARCHAR(64)   NOT NULL DEFAULT 'v1.0',
    -- Input conditions
    supply_temp_c   NUMERIC(6,3)  NOT NULL,
    flow_rate_lpm   NUMERIC(8,2)  NOT NULL,
    it_power_mw     NUMERIC(10,6) NOT NULL,
    -- Predicted thermal field (flattened 8x8 JSON array)
    thermal_field   JSONB         NOT NULL,
    -- Key aggregated metrics from field
    pred_max_inlet  NUMERIC(6,3),
    pred_mean_inlet NUMERIC(6,3),
    pred_hotspot_x  SMALLINT,
    pred_hotspot_y  SMALLINT,
    -- Ground truth (filled after actual measurement)
    actual_max_inlet NUMERIC(6,3),
    mae_c            NUMERIC(6,4),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY (facility_id, crac_id) REFERENCES crac_units(facility_id, crac_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_fno_facility ON fno_predictions(facility_id, predicted_at DESC);

-- =============================================================================
-- 7. Drift Detection Events
-- =============================================================================
CREATE TYPE drift_status AS ENUM ('DETECTED', 'ACKNOWLEDGED', 'RETRAINING_TRIGGERED', 'RESOLVED');

CREATE TABLE IF NOT EXISTS drift_events (
    event_id         UUID         PRIMARY KEY DEFAULT uuid_generate_v4(),
    facility_id      VARCHAR(64)  NOT NULL REFERENCES facilities(facility_id) ON DELETE CASCADE,
    detected_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    drift_metric     VARCHAR(64)  NOT NULL,
    drift_score      NUMERIC(8,6) NOT NULL,
    threshold        NUMERIC(8,6) NOT NULL,
    status           drift_status NOT NULL DEFAULT 'DETECTED',
    step_fn_execution_arn VARCHAR(512),
    resolved_at      TIMESTAMPTZ,
    metadata         JSONB        NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_drift_facility ON drift_events(facility_id, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_drift_status   ON drift_events(status);

-- =============================================================================
-- 8. Step Functions Execution Log
-- =============================================================================
CREATE TABLE IF NOT EXISTS orchestration_runs (
    run_id              UUID         PRIMARY KEY DEFAULT uuid_generate_v4(),
    state_machine_arn   VARCHAR(512) NOT NULL,
    execution_arn       VARCHAR(512) UNIQUE,
    facility_id         VARCHAR(64)  NOT NULL REFERENCES facilities(facility_id) ON DELETE CASCADE,
    trigger_event       VARCHAR(128) NOT NULL DEFAULT 'drift_detection',
    started_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    ended_at            TIMESTAMPTZ,
    status              VARCHAR(32)  NOT NULL DEFAULT 'RUNNING',
    steps_completed     INTEGER      NOT NULL DEFAULT 0,
    error_message       TEXT,
    input_payload       JSONB        NOT NULL DEFAULT '{}'::jsonb,
    output_payload      JSONB
);

CREATE INDEX IF NOT EXISTS idx_runs_facility ON orchestration_runs(facility_id, started_at DESC);

-- =============================================================================
-- 9. API Audit Log
-- =============================================================================
CREATE TABLE IF NOT EXISTS api_audit_log (
    log_id       BIGSERIAL    PRIMARY KEY,
    endpoint     VARCHAR(256) NOT NULL,
    method       VARCHAR(16)  NOT NULL,
    client_ip    INET,
    user_agent   TEXT,
    request_body JSONB,
    status_code  SMALLINT     NOT NULL,
    response_ms  INTEGER,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_time     ON api_audit_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_endpoint ON api_audit_log(endpoint, created_at DESC);

-- =============================================================================
-- 10. ASHRAE SLA Violation Ledger
-- =============================================================================
CREATE TABLE IF NOT EXISTS sla_violations (
    violation_id     UUID         PRIMARY KEY DEFAULT uuid_generate_v4(),
    facility_id      VARCHAR(64)  NOT NULL REFERENCES facilities(facility_id) ON DELETE CASCADE,
    crac_id          VARCHAR(64)  NOT NULL,
    rack_id          VARCHAR(64)  NOT NULL,
    violation_type   VARCHAR(32)  NOT NULL CHECK (violation_type IN ('BELOW_MIN', 'ABOVE_MAX', 'CRITICAL')),
    server_inlet_c   NUMERIC(6,3) NOT NULL,
    ashrae_limit_c   NUMERIC(5,1) NOT NULL,
    detected_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    duration_s       INTEGER,
    resolved_at      TIMESTAMPTZ,
    auto_corrected   BOOLEAN      NOT NULL DEFAULT FALSE,
    correction_action JSONB,
    FOREIGN KEY (facility_id, crac_id) REFERENCES crac_units(facility_id, crac_id) ON DELETE CASCADE,
    FOREIGN KEY (facility_id, rack_id) REFERENCES racks(facility_id, rack_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sla_facility ON sla_violations(facility_id, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_sla_type     ON sla_violations(violation_type, detected_at DESC);

-- =============================================================================
-- Helper: auto-update updated_at timestamp
-- =============================================================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER facilities_updated_at
    BEFORE UPDATE ON facilities
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER crac_units_updated_at
    BEFORE UPDATE ON crac_units
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- Seed data: 3 facilities × 4 CRACs × representative racks
-- Composite PKs mean the same crac_id value ('CRAC-01' etc.) is valid for
-- every facility — no collisions across DC-EAST-01, DC-WEST-02, DC-EU-01.
-- =============================================================================
INSERT INTO facilities (facility_id, facility_name, aws_region, nominal_capacity_mw, target_pue)
VALUES
    ('DC-EAST-01', 'East Coast Primary Data Center',  'us-east-1',   5.0, 1.15),
    ('DC-WEST-02', 'West Coast Secondary Data Center', 'us-west-2',   4.0, 1.18),
    ('DC-EU-01',   'EU Frankfurt Data Center',         'eu-central-1', 3.0, 1.20)
ON CONFLICT (facility_id) DO NOTHING;

-- ---------------------------------------------------------------------------
-- CRAC units: 4 per facility
-- ---------------------------------------------------------------------------
INSERT INTO crac_units (facility_id, crac_id, cooling_zone, cooling_capacity_kw, rated_flow_lpm)
VALUES
    -- DC-EAST-01
    ('DC-EAST-01', 'CRAC-01', 'north', 120.0, 7500.0),
    ('DC-EAST-01', 'CRAC-02', 'south', 120.0, 7500.0),
    ('DC-EAST-01', 'CRAC-03', 'east',  100.0, 6500.0),
    ('DC-EAST-01', 'CRAC-04', 'west',  100.0, 6500.0),
    -- DC-WEST-02
    ('DC-WEST-02', 'CRAC-01', 'north', 110.0, 7000.0),
    ('DC-WEST-02', 'CRAC-02', 'south', 110.0, 7000.0),
    ('DC-WEST-02', 'CRAC-03', 'east',   90.0, 6000.0),
    ('DC-WEST-02', 'CRAC-04', 'west',   90.0, 6000.0),
    -- DC-EU-01
    ('DC-EU-01', 'CRAC-01', 'north',  95.0, 6200.0),
    ('DC-EU-01', 'CRAC-02', 'south',  95.0, 6200.0),
    ('DC-EU-01', 'CRAC-03', 'east',   80.0, 5500.0),
    ('DC-EU-01', 'CRAC-04', 'west',   80.0, 5500.0)
ON CONFLICT (facility_id, crac_id) DO NOTHING;

-- ---------------------------------------------------------------------------
-- Racks: one representative rack per CRAC zone
-- ---------------------------------------------------------------------------
INSERT INTO racks (facility_id, rack_id, crac_id, grid_row, grid_col, max_thermal_rating_kw)
VALUES
    -- DC-EAST-01
    ('DC-EAST-01', 'RACK-A01', 'CRAC-01', 0, 0, 20.0),
    ('DC-EAST-01', 'RACK-A02', 'CRAC-01', 0, 1, 20.0),
    ('DC-EAST-01', 'RACK-B01', 'CRAC-01', 1, 0, 20.0),
    ('DC-EAST-01', 'RACK-E01', 'CRAC-02', 4, 0, 20.0),
    ('DC-EAST-01', 'RACK-E02', 'CRAC-02', 4, 1, 20.0),
    ('DC-EAST-01', 'RACK-A05', 'CRAC-03', 0, 4, 20.0),
    ('DC-EAST-01', 'RACK-A06', 'CRAC-03', 0, 5, 20.0),
    ('DC-EAST-01', 'RACK-E05', 'CRAC-04', 4, 4, 20.0),
    -- DC-WEST-02
    ('DC-WEST-02', 'RACK-A01', 'CRAC-01', 0, 0, 18.0),
    ('DC-WEST-02', 'RACK-A02', 'CRAC-01', 0, 1, 18.0),
    ('DC-WEST-02', 'RACK-E01', 'CRAC-02', 4, 0, 18.0),
    ('DC-WEST-02', 'RACK-E02', 'CRAC-02', 4, 1, 18.0),
    ('DC-WEST-02', 'RACK-A05', 'CRAC-03', 0, 4, 18.0),
    ('DC-WEST-02', 'RACK-E05', 'CRAC-04', 4, 4, 18.0),
    -- DC-EU-01
    ('DC-EU-01', 'RACK-A01', 'CRAC-01', 0, 0, 15.0),
    ('DC-EU-01', 'RACK-A02', 'CRAC-01', 0, 1, 15.0),
    ('DC-EU-01', 'RACK-E01', 'CRAC-02', 4, 0, 15.0),
    ('DC-EU-01', 'RACK-A05', 'CRAC-03', 0, 4, 15.0),
    ('DC-EU-01', 'RACK-E05', 'CRAC-04', 4, 4, 15.0)
ON CONFLICT (facility_id, rack_id) DO NOTHING;
