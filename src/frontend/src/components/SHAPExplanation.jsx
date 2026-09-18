import React from 'react';
import { Brain, HelpCircle, ShieldCheck, Sparkles } from 'lucide-react';

export default function SHAPExplanation({
  attributions = [
    { feature: 'IT Workload (kW)', value: 0.42, direction: 'increase', desc: 'Ramping compute load demands higher thermal extraction' },
    { feature: 'Ambient Dry-Bulb (°C)', value: 0.35, direction: 'increase', desc: 'Elevated outdoor air temp lowers chiller baseline COP' },
    { feature: 'Supply Water Temp (°C)', value: -0.24, direction: 'decrease', desc: 'Current 18.5°C supply provides healthy ASHRAE headroom' },
    { feature: 'Grid Carbon (gCO₂e)', value: -0.16, direction: 'decrease', desc: 'Moderate grid carbon prioritizes free-cooling valve split' },
    { feature: 'Coolant Flow (L/min)', value: 0.09, direction: 'increase', desc: 'Stable differential pressure across CDU manifold' },
  ],
  safetyCost = 0.018,
  costLimit = 0.050,
}) {
  const safetyHeadroomPct = Math.round(((costLimit - safetyCost) / costLimit) * 100);

  return (
    <div className="glass-panel rounded-2xl p-5 border border-slate-800 flex flex-col justify-between">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
            <Brain className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-100">Explainable AI (SHAP)</h3>
            <span className="text-[11px] text-slate-400">Safe-PPO Policy Feature Attributions</span>
          </div>
        </div>

        {/* Safety Constraint Badge */}
        <div className="flex items-center space-x-1.5 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-1 rounded-full text-xs text-emerald-400">
          <ShieldCheck className="w-3.5 h-3.5" />
          <span className="font-mono font-bold">V_cost: {safetyCost.toFixed(3)}</span>
          <span className="text-slate-500">|</span>
          <span className="text-[10px] text-slate-400">{safetyHeadroomPct}% Safety Headroom</span>
        </div>
      </div>

      {/* Feature Attribution Bar Charts */}
      <div className="space-y-2.5 my-2">
        {attributions.map((attr, idx) => {
          const isPositive = attr.value > 0;
          const absVal = Math.abs(attr.value);
          const barWidth = Math.min(100, Math.round(absVal * 180));

          return (
            <div key={idx} className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-300 font-medium">{attr.feature}</span>
                <span className={`font-mono font-bold ${isPositive ? 'text-cyan-400' : 'text-emerald-400'}`}>
                  {isPositive ? `+${attr.value.toFixed(2)}` : attr.value.toFixed(2)}
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
          <span>
            Safe-PPO prioritized higher CDU pump speed (+4.2%) to absorb row-level IT power spikes while shifting 35% of cooling to the free-air economizer to minimize grid carbon impact.
          </span>
        </div>
      </div>
    </div>
  );
}
