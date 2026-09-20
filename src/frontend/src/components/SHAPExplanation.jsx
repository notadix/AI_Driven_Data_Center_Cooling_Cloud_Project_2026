import React, { useState, useEffect } from 'react';
import { API_BASE } from '../config';
import { Brain, ShieldCheck, Sparkles, RefreshCw, AlertCircle } from 'lucide-react';

const FEATURE_META = {
  it_power_kw: { label: 'IT Workload (kW)', desc: 'Compute load thermal rejection demand' },
  ambient_c: { label: 'Ambient Dry-Bulb (°C)', desc: 'Outdoor temperature affecting chiller COP' },
  grid_carbon_gco2: { label: 'Grid Carbon (gCO₂e)', desc: 'Marginal grid emissions steering valve split' },
  supply_c: { label: 'Coolant Supply Temp (°C)', desc: 'Chilled water supply baseline headroom' },
  return_c: { label: 'Coolant Return Temp (°C)', desc: 'Thermal energy returned from rack CDU' },
  flow_lpm: { label: 'CDU Flow Rate (L/min)', desc: 'Coolant circulation mass volume' },
  server_inlet_c: { label: 'Server Inlet Temp (°C)', desc: 'Intake air thermal SLA compliance' },
  server_outlet_c: { label: 'Server Outlet Temp (°C)', desc: 'Exhaust air delta-T across server rows' },
  cooling_kw: { label: 'Cooling Power (kW)', desc: 'Auxiliary fan and pump electrical load' },
  pue: { label: 'Facility PUE', desc: 'Infrastructure power efficiency ratio' },
};

export default function SHAPExplanation({
  facilityId = 'DC-EAST-01',
  cracId = 'CRAC-01',
}) {
  const [attributions, setAttributions] = useState([]);
  const [safetyCost, setSafetyCost] = useState(0.0);
  const [costLimit, setCostLimit] = useState(0.05);
  const [available, setAvailable] = useState(true);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [rationale, setRationale] = useState('');

  useEffect(() => {
    let isMounted = true;

    async function fetchExplainability() {
      try {
        const url = `${API_BASE}/api/v1/control/explain/${facilityId}/${cracId}`;

        const res = await fetch(url);
        if (!res.ok) {
          if (isMounted) {
            setError(`HTTP ${res.status}`);
            setLoading(false);
          }
          return;
        }

        const json = await res.json();
        if (!isMounted) return;

        if (json.status === 'ok' && json.data && json.data.available) {
          const rawAttrs = json.data.attributions || {};
          const parsed = Object.entries(rawAttrs).map(([key, val]) => {
            const meta = FEATURE_META[key] || { label: key, desc: '' };
            return {
              key,
              feature: meta.label,
              desc: meta.desc,
              value: Number(val),
              direction: val > 0 ? 'increase' : 'decrease',
            };
          });

          // Sort by absolute attribution magnitude
          parsed.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

          setAttributions(parsed);
          setSafetyCost(Number(json.data.v_cost || 0.0));
          setCostLimit(Number(json.data.cost_limit || 0.05));
          setAvailable(true);
          setError(null);
          setLoading(false);

          // Build dynamic rationale based on top positive & negative drivers
          const topPos = parsed.find((p) => p.value > 0);
          const topNeg = parsed.find((p) => p.value < 0);

          let dynamicRationale = 'Safe-PPO closed-loop policy evaluating real-time thermal conditions.';
          if (topPos && topNeg) {
            dynamicRationale = `Safe-PPO policy prioritized ${topPos.feature} (+${topPos.value.toFixed(2)}) as the primary cooling driver while ${topNeg.feature} (${topNeg.value.toFixed(2)}) moderated chilled-water consumption within ASHRAE SLA limits.`;
          } else if (topPos) {
            dynamicRationale = `Safe-PPO policy ramped actuator capacity driven predominantly by ${topPos.feature} (+${topPos.value.toFixed(2)}).`;
          }
          setRationale(dynamicRationale);
        } else {
          // 'unavailable' (no checkpoint / incomplete telemetry) or any unexpected shape: never spin forever.
          setAvailable(false);
          setError(json.reason || 'Explanation unavailable');
          setLoading(false);
        }
      } catch (err) {
        if (isMounted) {
          setError(err.message || 'Connection error');
          setLoading(false);
        }
      }
    }

    // Initial fetch
    setLoading(true);
    fetchExplainability();

    // Periodic refresh every 2.5 seconds
    const interval = setInterval(fetchExplainability, 2500);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [facilityId, cracId]);

  const safetyHeadroomPct = Math.max(
    0,
    Math.min(100, Math.round(((costLimit - safetyCost) / costLimit) * 100))
  );

  return (
    <div className="glass-panel rounded-2xl p-5 border border-slate-800 flex flex-col justify-between">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
            <Brain className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center space-x-1.5">
              <h3 className="text-sm font-semibold text-slate-100">Explainable AI (SHAP)</h3>
              <span className="text-[10px] font-mono text-cyan-400 bg-cyan-500/10 px-1.5 py-0.2 rounded border border-cyan-500/20">
                {cracId}
              </span>
            </div>
            <span className="text-[11px] text-slate-400">Safe-PPO Gradient × Input Saliency</span>
          </div>
        </div>

        {/* Safety Constraint Badge */}
        <div className="flex items-center space-x-1.5 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-1 rounded-full text-xs text-emerald-400">
          <ShieldCheck className="w-3.5 h-3.5" />
          <span className="font-mono font-bold">V_cost: {safetyCost.toFixed(3)}</span>
          <span className="text-slate-500">|</span>
          <span className="text-[10px] text-slate-400">{safetyHeadroomPct}% Headroom</span>
        </div>
      </div>

      {/* Content State Handling */}
      {loading && attributions.length === 0 ? (
        <div className="py-8 flex flex-col items-center justify-center text-slate-500 space-y-2">
          <RefreshCw className="w-5 h-5 animate-spin text-cyan-400" />
          <span className="text-xs">Computing live policy attributions...</span>
        </div>
      ) : !available ? (
        <div className="py-6 flex flex-col items-center justify-center text-amber-400/90 space-y-2 bg-amber-500/5 rounded-xl border border-amber-500/20 p-4">
          <AlertCircle className="w-5 h-5" />
          <span className="text-xs text-center font-medium">
            {error || 'Safe-PPO checkpoint not loaded. PID baseline controller active.'}
          </span>
        </div>
      ) : (
        <>
          {/* Feature Attribution Bar Charts (Top 5 Influential) */}
          <div className="space-y-2.5 my-2">
            {attributions.slice(0, 5).map((attr) => {
              const isPositive = attr.value > 0;
              const absVal = Math.abs(attr.value);
              const barWidth = Math.min(100, Math.round(absVal * 180));

              return (
                <div key={attr.key} className="space-y-1 group" title={attr.desc}>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-slate-300 font-medium group-hover:text-cyan-300 transition-colors">
                      {attr.feature}
                    </span>
                    <span className={`font-mono font-bold ${isPositive ? 'text-cyan-400' : 'text-emerald-400'}`}>
                      {isPositive ? `+${attr.value.toFixed(3)}` : attr.value.toFixed(3)}
                    </span>
                  </div>

                  {/* Bidirectional Bar */}
                  <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden flex items-center relative">
                    {/* Center Divider */}
                    <div className="absolute left-1/2 top-0 bottom-0 w-[1px] bg-slate-700 z-10" />

                    {isPositive ? (
                      <div
                        className="h-full bg-gradient-to-r from-cyan-500 to-cyan-400 rounded-r-full ml-auto"
                        style={{
                          width: `${barWidth / 2}%`,
                          marginRight: `${50 - barWidth / 2}%`,
                        }}
                      />
                    ) : (
                      <div
                        className="h-full bg-gradient-to-l from-emerald-500 to-emerald-400 rounded-l-full"
                        style={{
                          width: `${barWidth / 2}%`,
                          marginLeft: `${50 - barWidth / 2}%`,
                        }}
                      />
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Natural Language Rationale Box */}
          <div className="mt-3 bg-slate-950/80 border border-slate-800 rounded-xl p-3 text-xs text-slate-300 flex items-start space-x-2.5">
            <Sparkles className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
            <div className="space-y-0.5 leading-relaxed">
              <span className="font-semibold text-slate-100">Decision Rationale: </span>
              <span className="text-slate-300">
                {rationale || 'Evaluating real-time spatiotemporal observations.'}
              </span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
