import React from 'react';
import { Leaf, Sun, Wind, Flame, Radio, BarChart3, AlertCircle } from 'lucide-react';

export default function CarbonTracker({
  carbonIntensity = 285.0,
  totalFacilityKw = 20930.0,
  region = 'us-east-1',
  valveSplitPct = null,
}) {
  // Compute hourly carbon emission in kg CO2e / hour
  const hourlyCarbonKg = (totalFacilityKw * (carbonIntensity / 1000.0)).toFixed(1);

  // Carbon status levels
  const getCarbonStatus = (val) => {
    if (val < 180) return { label: 'ULTRA CLEAN', color: 'text-emerald-400', bg: 'bg-emerald-500/10 border-emerald-500/30' };
    if (val < 300) return { label: 'CLEAN GRID', color: 'text-cyan-400', bg: 'bg-cyan-500/10 border-cyan-500/30' };
    if (val < 420) return { label: 'MODERATE', color: 'text-amber-400', bg: 'bg-amber-500/10 border-amber-500/30' };
    return { label: 'DIRTY GRID', color: 'text-red-400', bg: 'bg-red-500/10 border-red-500/30' };
  };

  const status = getCarbonStatus(carbonIntensity);

  // Fuel mix percentages based on current carbon intensity
  const solarPct = Math.max(5, Math.min(45, Math.round(35 - (carbonIntensity - 150) * 0.08)));
  const windPct = Math.max(10, Math.min(40, Math.round(30 - (carbonIntensity - 150) * 0.05)));
  const hydroNuclearPct = 25;
  const fossilPct = Math.max(5, 100 - solarPct - windPct - hydroNuclearPct);
  const totalRenewablePct = solarPct + windPct + hydroNuclearPct;

  return (
    <div className="glass-panel rounded-2xl p-5 border border-slate-800 flex flex-col justify-between">
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
            <Leaf className="w-4 h-4" />
          </div>
          <h3 className="text-sm font-semibold text-slate-100">Green Grid Carbon Tracker</h3>
        </div>
        <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${status.bg} ${status.color}`}>
          {status.label}
        </span>
      </div>

      {/* Main Carbon Intensity Readout */}
      <div className="grid grid-cols-2 gap-3 my-2">
        <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/60">
          <span className="text-[11px] text-slate-400 font-medium">Marginal Intensity</span>
          <div className="flex items-baseline space-x-1.5 mt-0.5">
            <span className="text-2xl font-mono font-black text-slate-100">{carbonIntensity.toFixed(0)}</span>
            <span className="text-xs text-slate-400 font-mono">gCO₂e/kWh</span>
          </div>
          <div className="flex items-center space-x-1 mt-1 text-[11px] text-slate-500">
            <Radio className="w-3 h-3 text-cyan-400 animate-pulse" />
            <span>Feed: {region}</span>
          </div>
        </div>

        <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/60">
          <span className="text-[11px] text-slate-400 font-medium">Emission Rate</span>
          <div className="flex items-baseline space-x-1.5 mt-0.5">
            <span className="text-2xl font-mono font-black text-emerald-400">{hourlyCarbonKg}</span>
            <span className="text-xs text-slate-400 font-mono">kg/hr</span>
          </div>
          <span className="text-[11px] text-slate-500 mt-1 block">
            {totalRenewablePct}% Zero-Carbon
          </span>
        </div>
      </div>

      {/* Generation Fuel Mix Segmented Bar */}
      <div className="mt-2 space-y-1.5">
        <div className="flex justify-between text-[11px]">
          <span className="text-slate-400">Regional Generation Mix</span>
          <span className="font-mono text-emerald-400 font-semibold">{totalRenewablePct}% Renewable</span>
        </div>

        {/* Multi-segment bar */}
        <div className="w-full h-2.5 rounded-full bg-slate-800 overflow-hidden flex">
          <div style={{ width: `${solarPct}%` }} className="bg-amber-400 transition-all duration-500" title={`Solar: ${solarPct}%`} />
          <div style={{ width: `${windPct}%` }} className="bg-cyan-400 transition-all duration-500" title={`Wind: ${windPct}%`} />
          <div style={{ width: `${hydroNuclearPct}%` }} className="bg-emerald-500 transition-all duration-500" title={`Hydro/Nuclear: ${hydroNuclearPct}%`} />
          <div style={{ width: `${fossilPct}%` }} className="bg-slate-600 transition-all duration-500" title={`Fossil Gas/Coal: ${fossilPct}%`} />
        </div>

        {/* Legend */}
        <div className="grid grid-cols-4 gap-1 text-[10px] text-slate-400 pt-1 text-center font-mono">
          <span className="text-amber-400">Solar {solarPct}%</span>
          <span className="text-cyan-400">Wind {windPct}%</span>
          <span className="text-emerald-400">Hydro {hydroNuclearPct}%</span>
          <span className="text-slate-400">Fossil {fossilPct}%</span>
        </div>
      </div>

      {/* Carbon-Aware Optimization Badge */}
      <div className="mt-3 bg-emerald-950/30 border border-emerald-500/20 rounded-xl p-2 flex items-center justify-between text-xs">
        <span className="text-emerald-300">Free-air (economizer) valve</span>
        <span className="font-mono font-bold text-emerald-400">{valveSplitPct != null ? `${Number(valveSplitPct).toFixed(0)}% open` : '--'}</span>
      </div>
    </div>
  );
}
