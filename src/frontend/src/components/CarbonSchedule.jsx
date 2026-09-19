import React, { useState, useEffect } from 'react';
import { CalendarClock, Leaf, AlertCircle } from 'lucide-react';
import { API_BASE } from '../config';

const W = 300;
const H = 90;

function scale(values, lo, hi) {
  const span = hi - lo || 1;
  return values.map((v) => H - 6 - ((v - lo) / span) * (H - 12));
}

function path(ys) {
  const step = W / (ys.length - 1);
  return ys.map((y, i) => `${i === 0 ? 'M' : 'L'}${(i * step).toFixed(1)},${y.toFixed(1)}`).join(' ');
}

export default function CarbonSchedule({ facilityId = 'DC-EAST-01', flexibleFraction = 0.2, maxDelayH = 8 }) {
  const [plan, setPlan] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setPlan(null);
    setError(null);

    const load = async () => {
      try {
        const url = `${API_BASE}/api/v1/optimization/carbon-plan/${facilityId}?flexible_fraction=${flexibleFraction}&max_delay_h=${maxDelayH}`;
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        if (!cancelled) {
          setPlan(json.data);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) setError(e.message || 'Connection error');
      }
    };

    load();
    const id = setInterval(load, 30000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [facilityId, flexibleFraction, maxDelayH]);

  const carbon = plan?.carbon_gco2_kwh;
  const base = plan?.baseline_kw;
  const shifted = plan?.shifted_kw;

  let chart = null;
  if (carbon && base && shifted) {
    const loLoad = Math.min(...base, ...shifted) * 0.95;
    const hiLoad = Math.max(...base, ...shifted) * 1.02;
    chart = (
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-24" role="img" aria-label="Carbon intensity and IT load by hour">
        <path d={path(scale(carbon, Math.min(...carbon) * 0.95, Math.max(...carbon) * 1.02))}
              fill="none" stroke="#f59e0b" strokeWidth="1.6" />
        <path d={path(scale(base, loLoad, hiLoad))}
              fill="none" stroke="#64748b" strokeWidth="1.4" strokeDasharray="4 3" />
        <path d={path(scale(shifted, loLoad, hiLoad))}
              fill="none" stroke="#22d3ee" strokeWidth="1.8" />
      </svg>
    );
  }

  return (
    <div className="glass-panel rounded-2xl p-5 border border-slate-800">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
            <CalendarClock className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-100">Carbon-Aware Schedule</h3>
            <span className="text-[11px] text-slate-400">
              {Math.round(flexibleFraction * 100)}% of load deferrable up to {maxDelayH} h
            </span>
          </div>
        </div>
        {plan && (
          <div className="flex items-center space-x-1.5 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-1 rounded-full text-xs text-emerald-400">
            <Leaf className="w-3.5 h-3.5" />
            <span className="font-mono font-bold">-{plan.emissions_reduction_pct.toFixed(1)}% CO₂</span>
          </div>
        )}
      </div>

      {error && !plan && (
        <div className="flex items-center space-x-1.5 text-xs text-red-300">
          <AlertCircle className="w-3.5 h-3.5" />
          <span>{error}</span>
        </div>
      )}
      {!plan && !error && <div className="text-xs text-slate-500">Planning…</div>}

      {chart}

      {plan && (
        <>
          <div className="flex items-center justify-between text-[10px] text-slate-400 mt-1">
            <span><span className="text-amber-400">━</span> grid gCO₂/kWh</span>
            <span><span className="text-slate-400">╍</span> baseline IT</span>
            <span><span className="text-cyan-400">━</span> shifted IT</span>
          </div>
          <div className="grid grid-cols-2 gap-2 mt-3 text-xs">
            <div className="bg-slate-950/60 rounded-xl p-2 border border-slate-800/60">
              <div className="text-slate-400">Moved energy</div>
              <div className="font-mono font-bold text-slate-100">{Math.round(plan.moved_energy_kwh).toLocaleString()} kWh</div>
            </div>
            <div className="bg-slate-950/60 rounded-xl p-2 border border-slate-800/60">
              <div className="text-slate-400">Daily emissions</div>
              <div className="font-mono font-bold text-slate-100">
                {Math.round(plan.shifted_emissions_kg).toLocaleString()} kg
                <span className="text-[10px] text-slate-500 font-normal"> (was {Math.round(plan.baseline_emissions_kg).toLocaleString()})</span>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
