import React, { useState, useEffect } from 'react';
import { LineChart, Cpu, AlertCircle } from 'lucide-react';
import { API_BASE } from '../config';

const W = 300;
const H = 70;

function sparkline(values) {
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  const step = W / (values.length - 1 || 1);
  return values
    .map((v, i) => `${i === 0 ? 'M' : 'L'}${(i * step).toFixed(1)},${(H - 6 - ((v - lo) / span) * (H - 12)).toFixed(1)}`)
    .join(' ');
}

export default function ForecastPanel({ facilityId = 'DC-EAST-01', cracId = 'CRAC-01' }) {
  const [load, setLoad] = useState(null);
  const [field, setField] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoad(null);
    setField(null);
    setError(null);

    const get = async (path) => {
      const res = await fetch(`${API_BASE}/api/v1/forecast/${path}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    };

    const refresh = async () => {
      try {
        const [l, f] = await Promise.all([
          get(`load/${facilityId}`),
          get(`thermal-field/${facilityId}/${cracId}`),
        ]);
        if (cancelled) return;
        setLoad(l.status === 'ok' ? l.data : null);
        setField(f.status === 'ok' ? f.data : null);
        setError(l.status === 'ok' || f.status === 'ok' ? null : l.reason || f.reason || 'Predictive models unavailable');
      } catch (e) {
        if (!cancelled) setError(e.message || 'Connection error');
      }
    };

    refresh();
    const id = setInterval(refresh, 5000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [facilityId, cracId]);

  const forecast = load?.forecast;

  return (
    <div className="glass-panel rounded-2xl p-5 border border-slate-800">
      <div className="flex items-center space-x-2 mb-3">
        <div className="p-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
          <LineChart className="w-4 h-4" />
        </div>
        <div>
          <h3 className="text-sm font-semibold text-slate-100">Predictive Layer</h3>
          <span className="text-[11px] text-slate-400">60-min load forecast · FNO thermal surrogate</span>
        </div>
      </div>

      {error && !load && !field && (
        <div className="flex items-center space-x-1.5 text-xs text-red-300">
          <AlertCircle className="w-3.5 h-3.5" />
          <span>{error}</span>
        </div>
      )}

      {forecast && (
        <>
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-16" role="img" aria-label="Forecast IT load">
            <path d={sparkline([load.current_it_kw, ...forecast.map((p) => p.it_kw)])}
                  fill="none" stroke="#32ADE6" strokeWidth="1.8" />
          </svg>
          <div className="grid grid-cols-3 gap-2 text-xs mt-1">
            <div className="bg-slate-950/60 rounded-xl p-2 border border-slate-800/60">
              <div className="text-slate-400">IT now</div>
              <div className="font-mono font-bold text-slate-100">{(load.current_it_kw / 1000).toFixed(2)} MW</div>
            </div>
            <div className="bg-slate-950/60 rounded-xl p-2 border border-slate-800/60">
              <div className="text-slate-400">IT +60 min</div>
              <div className="font-mono font-bold text-slate-100">{(forecast[forecast.length - 1].it_kw / 1000).toFixed(2)} MW</div>
            </div>
            <div className="bg-slate-950/60 rounded-xl p-2 border border-slate-800/60">
              <div className="text-slate-400">PUE +60 min</div>
              <div className="font-mono font-bold text-slate-100">{forecast[forecast.length - 1].pue.toFixed(3)}</div>
            </div>
          </div>
        </>
      )}

      {field && (
        <div className="flex items-center justify-between mt-3 bg-slate-950/60 rounded-xl p-2 border border-slate-800/60 text-xs">
          <div className="flex items-center space-x-1.5 text-slate-400">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span>FNO rack coolant {cracId}</span>
          </div>
          <div className="font-mono text-slate-100">
            {field.min_c.toFixed(1)}–{field.max_c.toFixed(1)} °C
            <span className="text-[10px] text-slate-500"> · {field.latency_ms.toFixed(1)} ms</span>
          </div>
        </div>
      )}
    </div>
  );
}
