// Pure helpers behind the dashboard (no React, no DOM) so they can be unit-tested with `node --test`.

export const SLA_MIN_C = 18.0;
export const SLA_MAX_C = 27.0;
export const CRITICAL_C = 32.0;
export const PUE_TARGET = 1.15;
const ROWS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];

// Same classification as the backend (src/backend/api/v1/telemetry.py::_ashrae_status).
export function statusForTemp(tempC) {
  if (tempC >= CRITICAL_C) return 'CRITICAL';
  if (tempC > SLA_MAX_C || tempC < SLA_MIN_C) return 'SLA_BREACH';
  return 'NORMAL';
}

// 8x8 racks RACK-A01..H08. Each rack belongs to one of four cooling zones (quadrants) and has a fixed positional
// offset (0 to +1.75 C). Rack power is a random PLACEHOLDER (22-30 kW): the simulator has no per-rack sensors.
export function generateInitialGrid(rng = Math.random) {
  const grid = [];
  for (let r = 0; r < 8; r++) {
    for (let c = 1; c <= 8; c++) {
      const baseTemp = 21.0 + r * 0.4 + c * 0.2;
      grid.push({
        rack_id: `RACK-${ROWS[r]}${c < 10 ? `0${c}` : c}`,
        row: r,
        col: c - 1,
        temp_c: Number(baseTemp.toFixed(1)),
        offset_c: Number((r * 0.15 + (c - 1) * 0.1).toFixed(2)),
        power_kw: Number((22.0 + rng() * 8.0).toFixed(1)),
        ashrae_status: statusForTemp(baseTemp),
        crac_id: r < 4 ? (c <= 4 ? 'CRAC-01' : 'CRAC-03') : (c <= 4 ? 'CRAC-02' : 'CRAC-04'),
      });
    }
  }
  return grid;
}

// Rack temperature = the zone's live inlet temperature + the rack's fixed offset.
export function applyInletToGrid(grid, cracId, inletC) {
  const inlet = Number(inletC);
  if (!cracId || !Number.isFinite(inlet)) return grid;
  return grid.map((node) => {
    if (node.crac_id !== cracId) return node;
    const temp = inlet + (node.offset_c ?? 0);
    return { ...node, temp_c: Number(temp.toFixed(1)), ashrae_status: statusForTemp(temp) };
  });
}

// Hall-scale facility totals: mean of the per-unit representative-rack readings (MW) x 1000 (kW) x zoneScale.
export function facilityAggregate(records, zoneScale) {
  const rows = (records || []).filter((r) => Number.isFinite(Number(r.it_power_mw)) && Number.isFinite(Number(r.cooling_power_mw)));
  if (rows.length === 0) return null;
  const mean = (k) => rows.reduce((s, r) => s + Number(r[k]), 0) / rows.length;
  const itKw = mean('it_power_mw') * 1000.0 * zoneScale;
  const coolingKw = mean('cooling_power_mw') * 1000.0 * zoneScale;
  return { itKw, coolingKw, pue: itKw > 0 ? Number(((itKw + coolingKw) / itKw).toFixed(4)) : null };
}

export function pueVsTargetPct(pue, target = PUE_TARGET) {
  return ((pue - target) / target) * 100;
}

export function coolingSharePct(itKw, coolingKw) {
  return (100 * coolingKw) / Math.max(1, itKw + coolingKw);
}

// Same bands as the Green Grid Carbon Tracker (gCO2/kWh).
export function carbonBand(gPerKwh) {
  if (gPerKwh < 180) return { label: 'Ultra clean grid', tone: 'ultra' };
  if (gPerKwh < 300) return { label: 'Clean grid', tone: 'clean' };
  if (gPerKwh < 420) return { label: 'Moderate grid', tone: 'moderate' };
  return { label: 'Dirty grid', tone: 'dirty' };
}

// Hourly emission rate in kg CO2e per hour.
export function emissionRateKgPerHr(totalFacilityKw, gPerKwh) {
  return totalFacilityKw * (gPerKwh / 1000.0);
}

// Share of racks currently inside the SLA band, or null when there are no racks.
export function slaOkPct(grid) {
  return grid.length ? (100 * grid.filter((n) => n.ashrae_status === 'NORMAL').length) / grid.length : null;
}

// Alarm de-duplication: raise an alarm when a unit ENTERS a breach, and again every renotifyMs while it persists,
// instead of one identical entry per telemetry message. `prev` is that unit's last state ({status, at}) or undefined.
export function alarmDecision(prev, status, nowMs, renotifyMs = 30000) {
  const breach = status === 'CRITICAL' || status === 'SLA_BREACH';
  if (!breach) return { raise: false, next: { status: 'NORMAL', at: nowMs } };
  const entering = !prev || prev.status === 'NORMAL' || prev.status !== status;
  const stale = prev && nowMs - prev.at >= renotifyMs;
  if (entering || stale) return { raise: true, next: { status, at: nowMs } };
  return { raise: false, next: prev };
}
