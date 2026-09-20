import React, { useState, useEffect, useRef } from 'react';
import { Sliders, ToggleLeft, ToggleRight, AlertOctagon, Check, RefreshCw, Lock, ShieldAlert } from 'lucide-react';
import { API_BASE } from '../config';

export default function OperatorControlPanel({
  facilityId = 'DC-EAST-01',
  onModeChange,
  onSubmitAction,
  telemetry = {},
  selectedCrac: selectedCracProp,
  onSelectCrac,
}) {
  // Controlled by the dashboard when it passes selectedCrac (so the AI explanation and forecast follow the same
  // unit); falls back to local state when used standalone.
  const [localCrac, setLocalCrac] = useState('CRAC-01');
  const selectedCrac = selectedCracProp ?? localCrac;
  const setSelectedCrac = (id) => { setLocalCrac(id); if (onSelectCrac) onSelectCrac(id); };
  // The real mode for `selectedCrac`, fetched from the backend -- NOT
  // derived from a shared `telemetry.mode` flag. That value used to come
  // from the parent, but it's a single global optimistic flag with no
  // per-CRAC identity and no connection to the backend's actual per-CRAC
  // mode store, so switching this panel's CRAC tab never reflected that
  // CRAC's real state (e.g. selecting a CRAC a different client had put
  // into manual mode would still show "AUTO (SAFE-PPO)").
  const [actualMode, setActualMode] = useState('auto');
  const [deltaSupply, setDeltaSupply] = useState(0.0);
  const [pumpSpeed, setPumpSpeed] = useState(telemetry.pump_speed_pct || 75.0);
  const [fanSpeed, setFanSpeed] = useState(telemetry.fan_speed_pct || 70.0);
  const [valveSplit, setValveSplit] = useState(telemetry.valve_split_pct || 40.0);
  const [submitting, setSubmitting] = useState(false);
  const [submittedFeedback, setSubmittedFeedback] = useState(null);
  const [feedbackError, setFeedbackError] = useState(false);
  const feedbackTimer = useRef(null);
  // Manual commands bypass the safety shield, so before Apply the backend predicts the zone inlet temperature the
  // command would produce (POST /control/preview) and the panel warns when it is unsafe.
  const [preview, setPreview] = useState(null);

  const isManual = actualMode === 'manual';

  useEffect(() => {
    let cancelled = false;
    const fetchMode = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/control/status/${facilityId}`);
        if (res.ok) {
          const json = await res.json();
          const crac = json.data?.cracs?.find((c) => c.crac_id === selectedCrac);
          if (crac && !cancelled) {
            setActualMode(crac.mode);
            // While the policy is in control, keep the sliders at the unit's live settings so that switching
            // to MANUAL and pressing Apply does not jump the actuators to arbitrary defaults.
            if (crac.mode === 'auto') {
              const live = (v, lo, hi, fallback) => (Number.isFinite(v) ? Math.min(hi, Math.max(lo, Math.round(v))) : fallback);
              setPumpSpeed((p) => live(crac.current_pump_pct, 35, 100, p));
              setFanSpeed((p) => live(crac.current_fan_pct, 30, 100, p));
              setValveSplit((p) => live(crac.current_valve_pct, 0, 100, p));
              setDeltaSupply(0.0);
            }
          }
        }
      } catch {
        // Backend unreachable -- keep the last known mode rather than reset it.
      }
    };
    fetchMode();
    const interval = setInterval(fetchMode, 3000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [selectedCrac, facilityId]);

  const isOk = (res) => res?.status === 'ok';

  const showFeedback = (text, error = false, ms = 3000) => {
    setFeedbackError(error);
    setSubmittedFeedback(text);
    // Cancel the previous message's timer, otherwise it clears this newer message early.
    if (feedbackTimer.current) clearTimeout(feedbackTimer.current);
    feedbackTimer.current = setTimeout(() => setSubmittedFeedback(null), ms);
  };

  useEffect(() => () => { if (feedbackTimer.current) clearTimeout(feedbackTimer.current); }, []);

  useEffect(() => {
    if (!isManual) { setPreview(null); return undefined; }
    let cancelled = false;
    const id = setTimeout(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/control/preview/${facilityId}/${selectedCrac}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ delta_supply_c: Number(deltaSupply), valve_split_pct: Number(valveSplit) }),
        });
        const json = await res.json();
        if (!cancelled) setPreview(res.ok && json.status === 'ok' ? json.data : null);
      } catch {
        if (!cancelled) setPreview(null);
      }
    }, 250);
    return () => { cancelled = true; clearTimeout(id); };
  }, [isManual, facilityId, selectedCrac, deltaSupply, valveSplit]);

  const previewTone = { ok: 'text-emerald-300 border-emerald-500/30 bg-emerald-500/10', warning: 'text-amber-300 border-amber-500/30 bg-amber-500/10', breach: 'text-red-300 border-red-500/40 bg-red-500/10' };
  const needsConfirm = preview && preview.safety_level !== 'ok';

  const errorText = (res) =>
    typeof res?.detail === 'string' ? res.detail : 'Request rejected by the backend';

  const handleModeToggle = async () => {
    const nextMode = isManual ? 'auto' : 'manual';
    const res = onModeChange ? await onModeChange(selectedCrac, nextMode) : { status: 'ok' };
    // Only reflect the new mode if the backend accepted it (the status poll
    // would otherwise flip it back a few seconds later).
    if (isOk(res)) {
      setActualMode(nextMode);
    } else {
      showFeedback(errorText(res), true);
    }
  };

  const handleSubmit = async () => {
    setSubmitting(true);
    setSubmittedFeedback(null);
    try {
      const payload = {
        delta_supply_c: Number(deltaSupply),
        pump_speed_pct: Number(pumpSpeed),
        fan_speed_pct: Number(fanSpeed),
        valve_split_pct: Number(valveSplit),
        source: 'manual',
        safety_status: 'NORMAL',
      };
      const res = onSubmitAction ? await onSubmitAction(selectedCrac, payload) : { status: 'ok' };
      if (isOk(res)) {
        showFeedback('Setpoints dispatched successfully!');
      } else {
        showFeedback(errorText(res), true);
      }
    } finally {
      setSubmitting(false);
    }
  };

  const triggerEmergencyCooling = async () => {
    // The auto-control loop re-applies its own action to every CRAC in
    // 'auto' mode every couple of seconds, which would silently undo the
    // emergency override -- take the CRAC out of auto first.
    if (!isManual && onModeChange) {
      const modeRes = await onModeChange(selectedCrac, 'manual');
      if (!isOk(modeRes)) {
        showFeedback(errorText(modeRes), true, 5000);
        return;
      }
      setActualMode('manual');
    }
    // Force maximum cooling override
    const payload = {
      delta_supply_c: -2.0,
      pump_speed_pct: 100.0,
      fan_speed_pct: 100.0,
      valve_split_pct: 0.0, // Full chilled water
      source: 'manual',
      safety_status: 'CRITICAL',
    };
    const res = onSubmitAction ? await onSubmitAction(selectedCrac, payload) : { status: 'ok' };
    if (isOk(res)) {
      showFeedback('EMERGENCY COOLING ACTIVATED (100% PUMP/FAN)', false, 6000);
    } else {
      showFeedback(errorText(res), true, 5000);
    }
  };

  return (
    <div className="glass-panel rounded-2xl p-5 border border-slate-800 flex flex-col justify-between">
      {/* Header & Mode Switch */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
            <Sliders className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-100">Supervisory Controls</h3>
            <span className="text-[11px] text-slate-400">Target Unit: {selectedCrac}</span>
          </div>
        </div>

        {/* Mode Toggle Button */}
        <button
          onClick={handleModeToggle}
          className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-xl border text-xs font-semibold transition-all ${
            isManual
              ? 'bg-amber-500/20 border-amber-500/40 text-amber-300 shadow-sm'
              : 'bg-emerald-500/20 border-emerald-500/40 text-emerald-300 shadow-sm'
          }`}
        >
          {isManual ? <ToggleRight className="w-4 h-4 text-amber-400" /> : <ToggleLeft className="w-4 h-4 text-emerald-400" />}
          <span>{isManual ? 'MANUAL OVERRIDE' : 'AUTO (SAFE-PPO)'}</span>
        </button>
      </div>

      {/* CRAC Unit Tabs */}
      <div className="grid grid-cols-4 gap-1.5 mb-3 bg-slate-950/60 p-1 rounded-xl border border-slate-800">
        {['CRAC-01', 'CRAC-02', 'CRAC-03', 'CRAC-04'].map((id) => (
          <button
            key={id}
            onClick={() => setSelectedCrac(id)}
            className={`py-1 rounded-lg text-xs font-mono transition-all ${
              selectedCrac === id
                ? 'bg-cyan-500/20 text-cyan-300 font-bold border border-cyan-500/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {id}
          </button>
        ))}
      </div>

      {/* Actuator Sliders Container */}
      <div className="space-y-3 relative">
        {!isManual && (
          <div className="absolute inset-0 bg-slate-950/70 backdrop-blur-[2px] rounded-xl flex items-center justify-center z-20 text-xs text-slate-300 space-x-2 border border-slate-800/80">
            <Lock className="w-4 h-4 text-cyan-400" />
            <span>Autonomous Safe-PPO in Control. Switch to Manual to adjust.</span>
          </div>
        )}

        {/* Delta Supply Temp */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-slate-400">Δ Supply Water Temp</span>
            <span className="font-mono font-bold text-slate-100">{deltaSupply > 0 ? `+${deltaSupply}` : deltaSupply}°C</span>
          </div>
          <input
            type="range"
            min="-2.0"
            max="2.0"
            step="0.1"
            disabled={!isManual}
            value={deltaSupply}
            onChange={(e) => setDeltaSupply(parseFloat(e.target.value))}
            className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
          />
        </div>

        {/* Pump Speed */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-slate-400">CDU Pump Speed</span>
            <span className="font-mono font-bold text-slate-100">{pumpSpeed}%</span>
          </div>
          <input
            type="range"
            min="35"
            max="100"
            step="1"
            disabled={!isManual}
            value={pumpSpeed}
            onChange={(e) => setPumpSpeed(parseFloat(e.target.value))}
            className="w-full accent-emerald-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
          />
        </div>

        {/* Fan Speed */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-slate-400">CRAC Air Fan Speed</span>
            <span className="font-mono font-bold text-slate-100">{fanSpeed}%</span>
          </div>
          <input
            type="range"
            min="30"
            max="100"
            step="1"
            disabled={!isManual}
            value={fanSpeed}
            onChange={(e) => setFanSpeed(parseFloat(e.target.value))}
            className="w-full accent-blue-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
          />
        </div>

        {/* Valve Split */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-slate-400">Free-Air / Chilled Water Split</span>
            <span className="font-mono font-bold text-slate-100">{valveSplit}% Free-Air</span>
          </div>
          <input
            type="range"
            min="0"
            max="100"
            step="5"
            disabled={!isManual}
            value={valveSplit}
            onChange={(e) => setValveSplit(parseFloat(e.target.value))}
            className="w-full accent-amber-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
          />
        </div>
      </div>

      {/* Feedback Message */}
      {submittedFeedback && (
        <div className={`mt-3 p-2 rounded-xl text-xs flex items-center space-x-1.5 border ${
          feedbackError
            ? 'bg-red-500/15 border-red-500/30 text-red-300'
            : 'bg-emerald-500/15 border-emerald-500/30 text-emerald-300'
        }`}>
          {feedbackError ? <AlertOctagon className="w-3.5 h-3.5" /> : <Check className="w-3.5 h-3.5" />}
          <span>{submittedFeedback}</span>
        </div>
      )}

      {isManual && preview && (
        <div className={`mt-3 text-[11px] rounded-lg border px-2.5 py-1.5 ${previewTone[preview.safety_level] || previewTone.ok}`}>
          <span className="font-semibold">Predicted zone inlet after Apply: {preview.predicted_inlet_c.toFixed(1)} °C.</span>{' '}
          {' '}It is {preview.detail}. Manual commands bypass the safety shield.
        </div>
      )}

      {/* Action Buttons */}
      <div className="grid grid-cols-2 gap-2 mt-4">
        <button
          onClick={handleSubmit}
          disabled={!isManual || submitting}
          className={`py-2 px-3 rounded-xl text-xs font-semibold flex items-center justify-center space-x-1.5 transition-all ${
            isManual
              ? 'bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold shadow-glow-cyan'
              : 'bg-slate-800 text-slate-500 cursor-not-allowed'
          }`}
        >
          {submitting ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Check className="w-3.5 h-3.5" />}
          <span>{needsConfirm ? 'Apply anyway' : 'Apply Setpoints'}</span>
        </button>

        {/* Emergency Kill-Switch */}
        <button
          onClick={triggerEmergencyCooling}
          className="py-2 px-3 rounded-xl text-xs font-bold bg-red-600/20 hover:bg-red-600/40 text-red-400 border border-red-500/40 flex items-center justify-center space-x-1.5 transition-all shadow-glow-red"
        >
          <AlertOctagon className="w-3.5 h-3.5 text-red-400" />
          <span>EMERGENCY KILL</span>
        </button>
      </div>
    </div>
  );
}
