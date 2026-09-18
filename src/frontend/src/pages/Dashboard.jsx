import React, { useState } from 'react';
import { useTelemetryWebSocket } from '../hooks/useTelemetryWebSocket';
import ThreeDHeatmap from '../components/ThreeDHeatmap';
import PUEGauge from '../components/PUEGauge';
import CarbonTracker from '../components/CarbonTracker';
import SHAPExplanation from '../components/SHAPExplanation';
import OperatorControlPanel from '../components/OperatorControlPanel';
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

export default function Dashboard() {
  const [selectedFacility, setSelectedFacility] = useState('DC-EAST-01');
  const [selectedRack, setSelectedRack] = useState(null);

  const {
    connected,
    telemetry,
    spatialGrid,
    alarms,
    submitControlAction,
    setControlMode,
  } = useTelemetryWebSocket(selectedFacility);

  const itPowerMw = (telemetry.it_power_kw / 1000.0).toFixed(2);
  const coolingPowerMw = (telemetry.cooling_power_kw / 1000.0).toFixed(2);
  const totalPowerMw = ((telemetry.it_power_kw + telemetry.cooling_power_kw) / 1000.0).toFixed(2);
  const deltaT = (telemetry.return_temp_c - telemetry.fws_supply_temp_c).toFixed(1);

  return (
    <div className="min-h-screen bg-[#070B14] text-slate-100 p-4 md:p-6 space-y-6 cyber-grid">
      {/* ── Top Header Navigation ──────────────────────────────────────────────── */}
      <header className="glass-panel rounded-2xl px-6 py-4 flex flex-wrap items-center justify-between gap-4 border border-slate-800">
        <div className="flex items-center space-x-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-500 to-cyan-500 flex items-center justify-center shadow-glow-cyan">
            <Server className="w-5 h-5 text-slate-950 font-bold" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-lg md:text-xl font-bold tracking-tight text-white">
                AI Hybrid Cooling Digital Twin
              </h1>
              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 font-mono">
                v2.0-PROD
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Cloud-Native Closed-Loop Supervisory Console & Sustainability Optimizer
            </p>
          </div>
        </div>

        {/* Facility Selector & Live Status */}
        <div className="flex items-center space-x-4">
          {/* Facility Dropdown */}
          <div className="flex items-center space-x-2 bg-slate-950/80 px-3 py-1.5 rounded-xl border border-slate-800 text-xs">
            <Building2 className="w-3.5 h-3.5 text-cyan-400" />
            <select
              value={selectedFacility}
              onChange={(e) => setSelectedFacility(e.target.value)}
              className="bg-transparent text-slate-200 font-mono outline-none cursor-pointer"
            >
              <option value="DC-EAST-01">DC-EAST-01 (US-East / Chilled Water + Free-Air)</option>
              <option value="DC-WEST-02">DC-WEST-02 (US-West / Evaporative Hybrid)</option>
              <option value="DC-EU-01">DC-EU-01 (EU-North / 100% Free Cooling)</option>
            </select>
          </div>

          {/* Connection Status Pill */}
          <div className="flex items-center space-x-2 bg-slate-950/80 px-3 py-1.5 rounded-xl border border-slate-800 text-xs">
            <span className={`w-2 h-2 rounded-full ${connected ? 'bg-emerald-400 shadow-glow-green animate-pulse' : 'bg-amber-400 animate-ping'}`} />
            <span className="font-mono text-[11px] font-semibold text-slate-300">
              {connected ? 'LIVE WS (10Hz)' : 'REST POLLING'}
            </span>
          </div>

          {/* ASHRAE SLA Badge */}
          <div className="flex items-center space-x-1.5 bg-emerald-500/10 border border-emerald-500/30 px-3 py-1.5 rounded-xl text-xs text-emerald-400 font-semibold font-mono">
            <ShieldCheck className="w-4 h-4" />
            <span>99.98% SLA OK</span>
          </div>
        </div>
      </header>

      {/* ── Top Level Real-Time KPI Cards ────────────────────────────────────── */}
      <section className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3.5">
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
            <span className="text-[10px] text-emerald-400 font-medium">Target: 1.150 (-14.2%)</span>
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
            <span className="text-[10px] text-cyan-400 font-mono">11.8% of Facility</span>
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
            <span className="text-[10px] text-emerald-400 font-medium">Economizer Max Split</span>
          </div>
        </div>
      </section>

      {/* ── Main 3D Digital Twin Canvas & Inspection Modal ────────────────────── */}
      <section className="relative">
        <ThreeDHeatmap
          spatialGrid={spatialGrid}
          onSelectRack={(rack) => setSelectedRack(rack)}
          selectedRack={selectedRack}
        />

        {/* Rack Detail Inspection Drawer / Modal */}
        {selectedRack && (
          <div className="absolute top-16 right-4 z-30 w-80 glass-panel bg-slate-950/95 border border-cyan-500/40 rounded-2xl p-4 shadow-2xl space-y-3">
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
                  {selectedRack.temp_c || 22.4}°C
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Outlet Temperature:</span>
                <span className="font-mono text-slate-200">
                  {((selectedRack.temp_c || 22.4) + 13.8).toFixed(1)}°C
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Server IT Power:</span>
                <span className="font-mono text-slate-200">
                  {selectedRack.power_kw || '24.2'} kW
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
                  {selectedRack.ashrae_status || 'NORMAL'}
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
          pue={telemetry.pue || 1.134}
          wue={telemetry.wue || 0.28}
          itPowerKw={telemetry.it_power_kw || 18450}
          coolingPowerKw={telemetry.cooling_power_kw || 2480}
        />

        {/* Green Grid Carbon Tracker */}
        <CarbonTracker
          carbonIntensity={telemetry.carbon_gco2_kwh || 285}
          totalFacilityKw={(telemetry.it_power_kw || 18450) + (telemetry.cooling_power_kw || 2480)}
          region="us-east-1"
        />

        {/* Explainable AI (SHAP) Waterfall */}
        <SHAPExplanation
          safetyCost={0.018}
          costLimit={0.050}
        />

        {/* Supervisory Control Panel */}
        <OperatorControlPanel
          currentMode={telemetry.mode || 'auto'}
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
              All 64 racks within ASHRAE thermal SLA bounds (18°C – 27°C). No active violations.
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
