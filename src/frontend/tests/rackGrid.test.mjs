import test from 'node:test';
import assert from 'node:assert/strict';
import {
  statusForTemp, generateInitialGrid, applyInletToGrid, facilityAggregate,
  pueVsTargetPct, coolingSharePct, carbonBand, emissionRateKgPerHr, slaOkPct,
} from '../src/utils/rackGrid.js';

test('status thresholds match the ASHRAE band and the backend', () => {
  assert.equal(statusForTemp(24), 'NORMAL');
  assert.equal(statusForTemp(18), 'NORMAL');
  assert.equal(statusForTemp(27), 'NORMAL');
  assert.equal(statusForTemp(27.01), 'SLA_BREACH');
  assert.equal(statusForTemp(17.99), 'SLA_BREACH');
  assert.equal(statusForTemp(32), 'CRITICAL');
});

test('grid has 64 uniquely named racks in four 16-rack zones', () => {
  const g = generateInitialGrid(() => 0.5);
  assert.equal(g.length, 64);
  assert.equal(new Set(g.map((n) => n.rack_id)).size, 64);
  assert.equal(g[0].rack_id, 'RACK-A01');
  assert.equal(g[63].rack_id, 'RACK-H08');
  const zones = {};
  g.forEach((n) => { zones[n.crac_id] = (zones[n.crac_id] || 0) + 1; });
  assert.deepEqual(zones, { 'CRAC-01': 16, 'CRAC-02': 16, 'CRAC-03': 16, 'CRAC-04': 16 });
  assert.ok(g.every((n) => n.power_kw >= 22 && n.power_kw <= 30));
  assert.equal(Math.max(...g.map((n) => n.offset_c)), 1.75);
});

test('zone inlet is applied to that zone only, plus each rack offset', () => {
  const g = applyInletToGrid(generateInitialGrid(() => 0), 'CRAC-02', 26.0);
  const z2 = g.filter((n) => n.crac_id === 'CRAC-02');
  assert.ok(z2.every((n) => Math.abs(n.temp_c - (26.0 + n.offset_c)) < 0.06));
  assert.ok(g.filter((n) => n.crac_id !== 'CRAC-02').every((n) => n.temp_c < 26));   // other zones untouched
});

test('a 26 C zone puts only the racks with a large enough offset above 27 C', () => {
  const g = applyInletToGrid(generateInitialGrid(() => 0), 'CRAC-04', 26.0);
  const zone = g.filter((n) => n.crac_id === 'CRAC-04');
  const breach = zone.filter((n) => n.ashrae_status === 'SLA_BREACH').length;
  assert.ok(breach > 0 && breach < zone.length);
  assert.ok(zone.every((n) => (n.ashrae_status === 'SLA_BREACH') === (n.temp_c > 27)));
});

test('bad inputs leave the grid unchanged instead of producing NaN colours', () => {
  const g = generateInitialGrid(() => 0);
  assert.equal(applyInletToGrid(g, 'CRAC-01', NaN), g);
  assert.equal(applyInletToGrid(g, 'CRAC-01', undefined), g);
  assert.equal(applyInletToGrid(g, null, 24), g);
});

test('facility totals are the mean of the units scaled to hall level', () => {
  const agg = facilityAggregate([
    { it_power_mw: 0.010, cooling_power_mw: 0.0004 }, { it_power_mw: 0.020, cooling_power_mw: 0.0008 },
  ], 1000);
  assert.ok(Math.abs(agg.itKw - 15000) < 1e-6);
  assert.ok(Math.abs(agg.coolingKw - 600) < 1e-6);
  assert.equal(agg.pue, 1.04);
  assert.equal(facilityAggregate([], 1000), null);
  assert.equal(facilityAggregate([{ it_power_mw: 'x', cooling_power_mw: 1 }], 1000), null);
});

test('PUE badge is relative to the 1.15 target and cooling share is computed', () => {
  assert.ok(Math.abs(pueVsTargetPct(1.05) - (-8.6957)) < 1e-3);
  assert.ok(pueVsTargetPct(1.3) > 0);
  assert.ok(Math.abs(coolingSharePct(21000, 1000) - 4.545) < 0.01);
  assert.equal(coolingSharePct(0, 0), 0);
});

test('carbon bands and emission rate', () => {
  assert.equal(carbonBand(150).tone, 'ultra');
  assert.equal(carbonBand(250).tone, 'clean');
  assert.equal(carbonBand(350).tone, 'moderate');
  assert.equal(carbonBand(525).label, 'Dirty grid');
  assert.ok(Math.abs(emissionRateKgPerHr(22150, 525) - 11628.75) < 0.01);
});

test('SLA percentage over the grid', () => {
  const g = generateInitialGrid(() => 0);
  assert.ok(slaOkPct(applyInletToGrid(g, 'CRAC-01', 26.5)) < 100);
  assert.equal(slaOkPct(applyInletToGrid(g, 'CRAC-01', 22)), 100);
  assert.equal(slaOkPct([]), null);
});

test('alarms are raised on entering a breach and then re-notified only every 30 s', async () => {
  const { alarmDecision } = await import('../src/utils/rackGrid.js');
  let st;
  let d = alarmDecision(st, 'NORMAL', 0); assert.equal(d.raise, false); st = d.next;
  d = alarmDecision(st, 'SLA_BREACH', 1000); assert.equal(d.raise, true); st = d.next;       // enters a breach
  for (let t = 2000; t < 30000; t += 1000) { d = alarmDecision(st, 'SLA_BREACH', t); assert.equal(d.raise, false); st = d.next; }
  d = alarmDecision(st, 'SLA_BREACH', 31000); assert.equal(d.raise, true); st = d.next;      // 30 s later: reminder
  d = alarmDecision(st, 'NORMAL', 32000); assert.equal(d.raise, false); st = d.next;         // recovered
  d = alarmDecision(st, 'SLA_BREACH', 33000); assert.equal(d.raise, true);                   // a NEW breach alarms at once
  d = alarmDecision({ status: 'SLA_BREACH', at: 0 }, 'CRITICAL', 500); assert.equal(d.raise, true);   // escalation
});
