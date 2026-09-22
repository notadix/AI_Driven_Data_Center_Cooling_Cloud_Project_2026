import React, { useCallback, useState } from 'react';
import { useTelemetryWebSocket } from '../hooks/useTelemetryWebSocket';
import ThreeDHeatmap from '../components/ThreeDHeatmap';
import PUEGauge from '../components/PUEGauge';
import CarbonTracker from '../components/CarbonTracker';
import SHAPExplanation from '../components/SHAPExplanation';
import OperatorControlPanel from '../components/OperatorControlPanel';
import CarbonSchedule from '../components/CarbonSchedule';
import ForecastPanel from '../components/ForecastPanel';
import OperatorLogin from '../components/OperatorLogin';
import { carbonBand, coolingSharePct as coolingShare, pueVsTargetPct, slaOkPct as slaShare } from '../utils/rackGrid';
import {
  Activity,
  Server,
  Cpu,
  Thermometer,
  Wind,
  ShieldCheck,
  Radio,
  Building2,
  Bell,
  X,
  Clock,
} from 'lucide-react';

// AWS region each simulated facility is deployed in (matches the climate/carbon
// profiles in the backend's REGIONAL_CLIMATE_BASE).
const FACILITY_REGIONS = {
  'DC-EAST-01': 'us-east-1',
  'DC-WEST-02': 'us-west-2',
  'DC-EU-01': 'eu-west-1',
};

export default function Dashboard() {
  const [selectedFacility, setSelectedFacility] = useState('DC-EAST-01');
  const [selectedRack, setSelectedRack] = useState(null);
  // The cooling unit that the controls, the AI explanation and the forecast all refer to. Clicking a rack
  // selects that rack's unit; the control tabs change it too, so the panels never disagree.
  const [selectedCrac, setSelectedCrac] = useState('CRAC-01');
  // Stable identity: ThreeDHeatmap's WebGL scene-setup effect depends on
  // this callback, so a fresh inline arrow function here (recreated every
  // Dashboard re-render, which happens on every telemetry tick) would tear
  // down and rebuild the entire Three.js renderer/scene each time --
  // exhausting the browser's WebGL context limit within seconds.
  const handleSelectRack = useCallback((rack) => {
    setSelectedRack(rack);
    if (rack && rack.crac_id) setSelectedCrac(rack.crac_id);
  }, []);

  const {
    connected,
    telemetry,
    spatialGrid,
    alarms,
    submitControlAction,
    setControlMode,
    offline,
    hasData,
  } = useTelemetryWebSocket(selectedFacility);

  // The drawer shows the selected rack's LIVE values: selectedRack is only a
  // snapshot taken at click time, so look the rack up in the current grid.
  const liveRack = selectedRack
    ? spatialGrid.find((n) => n.rack_id === selectedRack.rack_id) || selectedRack
    : null;
  // Rack air delta-T: use the facility's live outlet-inlet delta rather than a
  // fixed 13.8 C.
  const rackDeltaT =
    telemetry.server_outlet_temp_c != null && telemetry.server_inlet_temp_c != null
      ? telemetry.server_outlet_temp_c - telemetry.server_inlet_temp_c
      : 0;

  // Live SLA compliance: share of racks whose inlet is inside the ASHRAE envelope right now.
  const slaOkPct = slaShare(spatialGrid);
  const pueDeltaPct = pueVsTargetPct(telemetry.pue);

  const itPowerMw = (telemetry.it_power_kw / 1000.0).toFixed(2);
  const coolingPowerMw = (telemetry.cooling_power_kw / 1000.0).toFixed(2);
  const totalPowerMw = ((telemetry.it_power_kw + telemetry.cooling_power_kw) / 1000.0).toFixed(2);
  const deltaT = (telemetry.return_temp_c - telemetry.fws_supply_temp_c).toFixed(1);
  // Same bands as the Green Grid Carbon Tracker (gCO2/kWh).
  const carbonNow = telemetry.carbon_gco2_kwh || 285;
  const band = carbonBand(carbonNow);
  const bandClass = { ultra: 'text-emerald-400', clean: 'text-cyan-400', moderate: 'text-amber-400', dirty: 'text-red-400' }[band.tone];
  const coolingSharePct = coolingShare(telemetry.it_power_kw, telemetry.cooling_power_kw).toFixed(1);
  const racksOutside = spatialGrid.filter((n) => n.ashrae_status !== 'NORMAL').length;

  return (
    <div className="min-h-screen bg-black text-slate-100 p-4 md:p-6 space-y-6 cyber-grid">
      {/* ── Top Header Navigation ──────────────────────────────────────────────── */}
      <header className="glass-panel rounded-2xl px-6 py-4 flex flex-wrap items-center justify-between gap-4 border border-slate-800 relative z-30">
        {/* relative z-30: header's own backdrop-filter creates a stacking context, so without an
            explicit z-index here the OperatorLogin popover (z-50, but scoped to that context) paints
            *behind* the KPI cards below, which have their own glass-panel stacking contexts and come
            later in DOM order. */}
        <div className="flex items-center space-x-4">
          <div className="relative w-10 h-10 rounded-[11px] bg-gradient-to-b from-emerald-400 via-emerald-500 to-cyan-500 flex items-center justify-center shadow-glow-cyan border border-white/25 overflow-hidden">
            {/* Glass sheen across the upper half, the way an Apple app icon catches light */}
            <div className="absolute inset-0 bg-gradient-to-b from-white/40 via-white/5 to-transparent pointer-events-none" />
            <div className="absolute inset-0 rounded-[11px] shadow-[inset_0_1px_1px_0_rgba(255,255,255,0.5),inset_0_-6px_10px_-6px_rgba(0,0,0,0.35)] pointer-events-none" />
            <Server className="relative w-5 h-5 text-white drop-shadow-sm" strokeWidth={2.25} />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-lg md:text-xl font-bold tracking-tight text-white">
                AI Hybrid Cooling Digital Twin
              </h1>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 font-mono whitespace-nowrap">
                v2.0-PROD
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Cloud-Native Closed-Loop Supervisory Console & Sustainability Optimizer
            </p>
          </div>
        </div>

        {/* Facility Selector & Live Status */}
        <div className="flex flex-wrap items-center gap-2 sm:gap-4">
          {/* Facility Dropdown */}
          <div className="flex items-center space-x-2 bg-slate-950/80 px-3 py-1.5 rounded-xl border border-slate-800 text-xs max-w-full">
            <Building2 className="w-3.5 h-3.5 text-cyan-400" />
            <select
              value={selectedFacility}
              onChange={(e) => {
                setSelectedFacility(e.target.value);
                setSelectedRack(null); // a rack from the previous facility must not stay selected
              }}
              className="bg-transparent text-slate-200 font-mono outline-none cursor-pointer min-w-0 max-w-[12.5rem] sm:max-w-none truncate"
            >
              <option value="DC-EAST-01">DC-EAST-01 (US-East / Chilled Water + Free-Air)</option>
              <option value="DC-WEST-02">DC-WEST-02 (US-West / Evaporative Hybrid)</option>
              <option value="DC-EU-01">DC-EU-01 (EU-North / 100% Free Cooling)</option>
            </select>
          </div>

          {/* Connection Status Pill */}
          <div className="flex items-center space-x-2 bg-slate-950/80 px-3 py-1.5 rounded-xl border border-slate-800 text-xs">
            <span className={`w-2 h-2 rounded-full ${connected ? 'bg-emerald-400 shadow-glow-green animate-pulse' : (offline ? 'bg-red-500' : 'bg-amber-400 animate-ping')}`} />
            <span className="font-mono text-[11px] font-semibold text-slate-300">
              {connected ? 'LIVE WS' : (offline ? 'BACKEND OFFLINE' : 'REST POLLING')}
            </span>
          </div>

          {/* ASHRAE SLA Badge */}
          <div className="flex items-center space-x-1.5 bg-emerald-500/10 border border-emerald-500/30 px-3 py-1.5 rounded-xl text-xs text-emerald-400 font-semibold font-mono">
            <ShieldCheck className="w-4 h-4" />
            <span>{slaOkPct === null ? 'SLA --' : `${slaOkPct.toFixed(1)}% racks in SLA`}</span>
          </div>

          <OperatorLogin />
        </div>
      </header>

      {/* ── Top Level Real-Time KPI Cards ────────────────────────────────────── */}
      <section className={`grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3.5 transition-opacity ${hasData ? '' : 'opacity-40'}`} title={hasData ? undefined : 'Waiting for the first telemetry reading'}>
        {/* PUE Card */}
        <div className="glass-panel p-4 rounded-2xl border border-slate-800 flex flex-col justify-between">
          <div className="flex justify-between items-start text-xs text-slate-400">
            <span>Facility PUE</span>
            <Activity className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="mt-2">
            <div className="text-2xl font-black font-mono text-slate-100">
              {telemetry.pue?.toFixed(3)}
            </div>
            <span className={`text-[10px] font-medium ${pueDeltaPct <= 0 ? 'text-emerald-400' : 'text-amber-400'}`}>
              Target: 1.150 ({pueDeltaPct <= 0 ? '' : '+'}{pueDeltaPct.toFixed(1)}%)
            </span>
          </div>
        </div>

        {/* IT Power Card */}
        <div className="glass-panel p-4 rounded-2xl border border-slate-800 flex flex-col justify-between">
          <div className="flex justify-between items-start text-xs text-slate-400">
            <span>IT Compute Load</span>
            <Cpu className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="mt-2">
            <div className="text-2xl font-black font-mono text-slate-100">
              {itPowerMw} <span className="text-xs font-normal text-slate-400">MW</span>
            </div>
            <span className="text-[10px] text-slate-400 font-mono">64 Server Rows</span>
          </div>
        </div>

        {/* Cooling Overhead */}
        <div className="glass-panel p-4 rounded-2xl border border-slate-800 flex flex-col justify-between">
          <div className="flex justify-between items-start text-xs text-slate-400">
            <span>Cooling Power</span>
            <Wind className="w-4 h-4 text-blue-400" />
          </div>
          <div className="mt-2">
            <div className="text-2xl font-black font-mono text-slate-100">
              {coolingPowerMw} <span className="text-xs font-normal text-slate-400">MW</span>
            </div>
            <span className="text-[10px] text-cyan-400 font-mono">{coolingSharePct}% of Facility</span>
          </div>
        </div>

        {/* Thermal Delta-T */}
        <div className="glass-panel p-4 rounded-2xl border border-slate-800 flex flex-col justify-between">
          <div className="flex justify-between items-start text-xs text-slate-400">
            <span>Water ΔT (Supply/Ret)</span>
            <Thermometer className="w-4 h-4 text-amber-400" />
          </div>
          <div className="mt-2">
            <div className="text-2xl font-black font-mono text-slate-100">
              {deltaT}°C
            </div>
            <span className="text-[10px] text-slate-400 font-mono">
              {telemetry.fws_supply_temp_c}°C → {telemetry.return_temp_c}°C
            </span>
          </div>
        </div>

        {/* CDU Flow Rate */}
        <div className="glass-panel p-4 rounded-2xl border border-slate-800 flex flex-col justify-between">
          <div className="flex justify-between items-start text-xs text-slate-400">
            <span>Coolant Flow Rate</span>
            <Activity className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="mt-2">
            <div className="text-2xl font-black font-mono text-slate-100">
              {Math.round(telemetry.flow_rate_lpm || 4850)} <span className="text-xs font-normal text-slate-400">LPM</span>
            </div>
            <span className="text-[10px] text-emerald-400 font-mono">Pump: {telemetry.pump_speed_pct || 74}%</span>
          </div>
        </div>

        {/* Carbon Intensity */}
        <div className="glass-panel p-4 rounded-2xl border border-slate-800 flex flex-col justify-between">
          <div className="flex justify-between items-start text-xs text-slate-400">
            <span>Grid Carbon</span>
            <Radio className="w-4 h-4 text-purple-400" />
          </div>
          <div className="mt-2">
            <div className="text-2xl font-black font-mono text-slate-100">
              {Math.round(telemetry.carbon_gco2_kwh || 285)} <span className="text-xs font-normal text-slate-400">g/kWh</span>
            </div>
            <span className={`text-[10px] font-medium ${bandClass}`}>{band.label}</span>
          </div>
        </div>
      </section>

      {/* ── Main 3D Digital Twin Canvas & Inspection Modal ────────────────────── */}
      <section className="relative">
        <ThreeDHeatmap
          spatialGrid={spatialGrid}
          onSelectRack={handleSelectRack}
          selectedRack={selectedRack}
        />

        {/* Rack Detail Inspection Drawer / Modal */}
        {selectedRack && (
          <div className="absolute top-40 sm:top-28 right-4 z-30 w-80 max-w-[calc(100%-2rem)] glass-panel bg-slate-950/95 border border-cyan-500/40 rounded-2xl p-4 shadow-2xl space-y-3">
            {/* The HUD row above (spatial-twin badge + camera switcher) can wrap onto two or
                three lines at narrow widths, so a small fixed offset overlaps those buttons. */}
            <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
              <div className="flex items-center space-x-2 font-mono font-bold text-slate-100">
                <Server className="w-4 h-4 text-cyan-400" />
                <span>{selectedRack.rack_id || selectedRack.rackId}</span>
              </div>
              <button
                onClick={() => setSelectedRack(null)}
                className="text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-slate-400">Inlet Temperature:</span>
                <span className="font-mono font-bold text-emerald-400">
                  {liveRack.temp_c ?? 22.4}°C
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Outlet Temperature:</span>
                <span className="font-mono text-slate-200">
                  {((liveRack.temp_c ?? 22.4) + rackDeltaT).toFixed(1)}°C
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Server IT Power:</span>
                <span className="font-mono text-slate-200">
                  {liveRack.power_kw ?? 24.2} kW
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Serving CRAC Unit:</span>
                <span className="font-mono text-cyan-400">
                  {selectedRack.crac_id || 'CRAC-01'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">ASHRAE TC 9.9 Status:</span>
                <span className="font-mono text-emerald-400 font-bold">
                  {liveRack.ashrae_status || 'NORMAL'}
                </span>
              </div>
            </div>

            <div className="pt-2 border-t border-slate-800">
              <span className="text-[11px] text-slate-500 block">
                Bound to AWS IoT SiteWise asset: <code className="text-cyan-400 font-mono">r-{(selectedRack.rack_id || selectedRack.rackId).toLowerCase()}</code>
              </span>
            </div>
          </div>
        )}
      </section>

      {/* ── Analytics & Controls 4-Column Grid ────────────────────────────────── */}
      <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* PUE & WUE Dial */}
        <PUEGauge
          valveSplitPct={telemetry.valve_split_pct}
          pue={telemetry.pue || 1.134}
          wue={telemetry.wue || 0.28}
          itPowerKw={telemetry.it_power_kw || 18450}
          coolingPowerKw={telemetry.cooling_power_kw || 2480}
        />

        {/* Green Grid Carbon Tracker */}
        <CarbonTracker
          valveSplitPct={telemetry.valve_split_pct}
          carbonIntensity={telemetry.carbon_gco2_kwh || 285}
          totalFacilityKw={(telemetry.it_power_kw || 18450) + (telemetry.cooling_power_kw || 2480)}
          region={FACILITY_REGIONS[selectedFacility] || 'us-east-1'}
        />

        {/* Carbon-aware workload schedule */}
        <CarbonSchedule facilityId={selectedFacility} />

        {/* Predictive layer: load forecast + FNO thermal surrogate */}
        <ForecastPanel facilityId={selectedFacility} cracId={selectedCrac} />

        {/* Explainable AI (SHAP) Waterfall */}
        <SHAPExplanation
          facilityId={selectedFacility}
          cracId={selectedCrac}
        />

        {/* Supervisory Control Panel */}
        <OperatorControlPanel
          facilityId={selectedFacility}
          selectedCrac={selectedCrac}
          onSelectCrac={setSelectedCrac}
          onModeChange={setControlMode}
          onSubmitAction={submitControlAction}
          telemetry={telemetry}
        />
      </section>

      {/* ── Real-Time SLA Alarms & Audit Log ─────────────────────────────────── */}
      <footer className="glass-panel rounded-2xl p-4 border border-slate-800">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2 text-xs font-semibold text-slate-200">
            <Bell className="w-4 h-4 text-cyan-400" />
            <span>Closed-Loop Telemetry & SLA Audit Log</span>
          </div>
          <span className="text-[11px] text-slate-500 font-mono">
            {alarms.length} Alarms / Events Recorded
          </span>
        </div>

        <div className="space-y-1.5 max-h-24 overflow-y-auto pr-1">
          {alarms.length === 0 ? (
            <div className="text-xs text-slate-500 italic py-1">
              {racksOutside === 0
                ? `All ${spatialGrid.length} racks within ASHRAE thermal SLA bounds (18°C – 27°C). No active violations.`
                : `${racksOutside} of ${spatialGrid.length} racks outside 18°C – 27°C. Rack temperatures are the live zone inlet plus a fixed illustrative per-rack offset; the controller enforces the limit on the zone inlet.`}
            </div>
          ) : (
            alarms.map((alarm) => (
              <div
                key={alarm.id}
                className="flex items-center justify-between text-xs p-2 rounded-lg bg-red-500/10 border border-red-500/20 text-red-300"
              >
                <div className="flex items-center space-x-2">
                  <span className="font-mono text-[10px] text-slate-400">{alarm.timestamp}</span>
                  <span>{alarm.message}</span>
                </div>
                <span className="font-mono font-bold text-[10px] bg-red-500/20 px-1.5 py-0.5 rounded">
                  {alarm.severity}
                </span>
              </div>
            ))
          )}
        </div>
      </footer>
    </div>
  );
}
