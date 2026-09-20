import React from 'react';
import { Zap, Droplets, Target, TrendingDown, ArrowDownRight } from 'lucide-react';

export default function PUEGauge({ pue = 1.134, wue = 0.28, itPowerKw = 18450, coolingPowerKw = 2480, valveSplitPct }) {
  // PUE Gauge Angle Calculation (scale from 1.0 to 1.6)
  // 1.0 = 0%, 1.6 = 100%
  const pueMin = 1.0;
  const pueMax = 1.6;
  const clampedPue = Math.min(Math.max(pue, pueMin), pueMax);
  const puePercent = (clampedPue - pueMin) / (pueMax - pueMin);
  
  // Circumference of radial circle (r = 52)
  const radius = 52;
  const strokeWidth = 10;
  const circumference = Math.PI * radius; // Half circle (180 deg)
  const strokeDashoffset = circumference - (circumference * puePercent);

  // Target indicator position (PUE 1.15)
  const targetPercent = (1.15 - pueMin) / (pueMax - pueMin);
  const targetAngleDeg = 180 * targetPercent; // 0 to 180

  // Status color
  const getPueColor = (val) => {
    if (val <= 1.15) return '#10B981'; // Emerald (Target achieved)
    if (val <= 1.30) return '#F59E0B'; // Amber
    return '#EF4444'; // Red
  };

  const currentColor = getPueColor(pue);

  return (
    <div className="glass-panel rounded-2xl p-5 border border-slate-800 flex flex-col justify-between">
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
            <Zap className="w-4 h-4" />
          </div>
          <h3 className="text-sm font-semibold text-slate-100">Efficiency Gauges</h3>
        </div>
        <div className="flex items-center space-x-1.5 text-xs text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20">
          <TrendingDown className="w-3.5 h-3.5" />
          <span className="font-mono font-medium">{`${(((pue - 1.15) / 1.15) * 100).toFixed(1)}% vs target`}</span>
        </div>
      </div>

      {/* Main Radial Meter */}
      <div className="flex flex-col items-center justify-center my-2 relative">
        <svg className="w-48 h-28 overflow-visible" viewBox="0 0 120 70">
          {/* Background Arc */}
          <path
            d="M 10,65 A 50,50 0 0,1 110,65"
            fill="none"
            stroke="#1E293B"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />

          {/* Active PUE Value Arc */}
          <path
            d="M 10,65 A 50,50 0 0,1 110,65"
            fill="none"
            stroke={currentColor}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            className="transition-all duration-700 ease-out"
          />

          {/* Target 1.15 Notch (Dotted pin) */}
          <circle
            cx={60 - 50 * Math.cos((targetAngleDeg * Math.PI) / 180)}
            cy={65 - 50 * Math.sin((targetAngleDeg * Math.PI) / 180)}
            r="3"
            fill="#38BDF8"
          />
        </svg>

        {/* Center Readout */}
        <div className="absolute top-10 flex flex-col items-center">
          <span className="text-3xl font-black tracking-tight font-mono text-slate-100">
            {pue.toFixed(3)}
          </span>
          <span className="text-[11px] font-semibold tracking-wider text-slate-400 uppercase">
            Facility PUE
          </span>
        </div>

        {/* Target Legend */}
        <div className="flex items-center space-x-1.5 text-[11px] text-slate-400 mt-1">
          <Target className="w-3 h-3 text-cyan-400" />
          <span>Target PUE:</span>
          <span className="font-mono text-cyan-300 font-bold">1.150</span>
        </div>
      </div>

      {/* Auxiliary Metrics Grid */}
      <div className="grid grid-cols-2 gap-3 pt-3 border-t border-slate-800/80">
        {/* WUE Dial Metric */}
        <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/60 flex items-center justify-between">
          <div>
            <div className="flex items-center space-x-1 text-[11px] text-slate-400">
              <Droplets className="w-3.5 h-3.5 text-blue-400" />
              <span>WUE</span>
            </div>
            <div className="font-mono font-bold text-base text-slate-100">
              {wue.toFixed(2)} <span className="text-[10px] text-slate-400 font-normal">L/kWh</span>
            </div>
          </div>
          <span className="text-[10px] text-emerald-400 font-mono bg-emerald-500/10 px-1.5 py-0.5 rounded">
            {valveSplitPct != null ? `Valve ${Number(valveSplitPct).toFixed(0)}%` : 'Valve --'}
          </span>
        </div>

        {/* Cooling Overhead */}
        <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/60">
          <span className="text-[11px] text-slate-400">Cooling Ratio</span>
          <div className="font-mono font-bold text-base text-cyan-300">
            {((coolingPowerKw / (itPowerKw + coolingPowerKw)) * 100).toFixed(1)}%
          </div>
        </div>
      </div>
    </div>
  );
}
