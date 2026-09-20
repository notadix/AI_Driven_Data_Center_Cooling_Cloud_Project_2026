import { useState, useEffect, useRef, useCallback } from 'react';
import { API_BASE, WS_BASE, ZONE_SCALE } from '../config';
import { generateInitialGrid, applyInletToGrid, facilityAggregate, alarmDecision } from '../utils/rackGrid';

/**
 * Resilient custom React hook for high-frequency telemetry streaming.
 * Features:
 * - Live WebSocket connection with exponential backoff reconnect
 * - Automatic heartbeat handling
 * - Fallback to REST polling if WebSocket is disconnected
 * - Built-in fallback simulator if backend is offline during standalone UI previews
 */
export function useTelemetryWebSocket(facilityId = 'DC-EAST-01') {
  const [connected, setConnected] = useState(false);
  // True once REST polling also fails: the backend is unreachable and the values shown are stale.
  const [offline, setOffline] = useState(false);
  // False until the first real reading arrives: until then `telemetry` holds placeholder values.
  const [hasData, setHasData] = useState(false);
  // Synthetic jitter is only for standalone UI demos (?demo=1); it must never pass for live data.
  const demoMode = typeof window !== 'undefined' && new URLSearchParams(window.location.search).get('demo') === '1';
  const [telemetry, setTelemetry] = useState({
    server_inlet_temp_c: 22.4,
    server_outlet_temp_c: 35.8,
    fws_supply_temp_c: 18.5,
    return_temp_c: 29.4,
    flow_rate_lpm: 4850.0,
    pump_speed_pct: 74.0,
    fan_speed_pct: 68.0,
    valve_split_pct: 35.0,
    it_power_kw: 18450.0,
    cooling_power_kw: 2480.0,
    pue: 1.134,
    wue: 0.28,
    carbon_gco2_kwh: 285.0,
    ashrae_status: 'NORMAL',
    mode: 'auto',
  });

  const [spatialGrid, setSpatialGrid] = useState(() => generateInitialGrid());
  const [cracStatus, setCracStatus] = useState([
    { crac_id: 'CRAC-01', zone: 'North', supply_c: 18.2, pump_pct: 75, status: 'NORMAL' },
    { crac_id: 'CRAC-02', zone: 'South', supply_c: 18.6, pump_pct: 72, status: 'NORMAL' },
    { crac_id: 'CRAC-03', zone: 'East',  supply_c: 18.4, pump_pct: 76, status: 'NORMAL' },
    { crac_id: 'CRAC-04', zone: 'West',  supply_c: 18.5, pump_pct: 74, status: 'NORMAL' },
  ]);

  const [alarms, setAlarms] = useState([]);
  // Latest per-CRAC power reading, keyed by crac_id. it_power_kw/
  // cooling_power_kw in `telemetry` represent the whole facility (the KPI
  // cards say "64 Server Rows"), but each simulated CRAC only reports its
  // own single representative rack's realistic ~10-28kW load (matching
  // the DB schema's per-rack max_thermal_rating_kw and the SiteWise/scene
  // per-rack bindings) -- with 4 CRACs now actively simulated, the facility
  // total must be the SUM of their readings, not whichever one's message
  // happened to arrive last (which made the KPI cards flicker between
  // ~0.01-0.03 MW, one CRAC's reading at a time, instead of a stable
  // facility aggregate).
  const cracPowerRef = useRef({});
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttempts = useRef(0);
  const pollIntervalRef = useRef(null);
  // Mirrors `connected` for the poll interval's closure without being a
  // dependency of the connect/poll effect below -- see its comment.
  const connectedRef = useRef(false);
  useEffect(() => {
    connectedRef.current = connected;
  }, [connected]);
  // Monotonic counter for alarm React keys: Date.now() alone collides when
  // more than one ASHRAE breach lands in the same millisecond, which is
  // routine at the 10Hz WebSocket rate (observed live as a duplicate-key
  // warning once telemetry actually started streaming after the connect
  // loop fix above).
  const alarmIdCounter = useRef(0);
  const alarmStateRef = useRef({});   // per cooling unit: last breach state, so a persistent breach is not logged every second

  // Connect WebSocket
  const connectWs = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;

    const wsUrl = `${WS_BASE}/ws/stream?facility_id=${facilityId}`;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      // Every handler below checks `wsRef.current === ws` first. Calling
      // .close() on a socket still fires ITS OWN onclose asynchronously
      // later; without this guard, switching facilityId (or unmounting)
      // would leave the old socket's onclose scheduling its own reconnect
      // via the stale closure's connectWs (bound to the OLD facilityId),
      // producing a "ghost" connection for an abandoned facility that
      // lives forever alongside the new one -- observed live via a
      // WebSocket traffic interceptor as two sockets (for two different,
      // no-longer-selected facilities) both still receiving heartbeats
      // indefinitely after switching facilities twice.
      ws.onopen = () => {
        if (wsRef.current !== ws) return;
        setConnected(true);
        setOffline(false);
        reconnectAttempts.current = 0;
      };

      ws.onmessage = (evt) => {
        if (wsRef.current !== ws) return;
        try {
          const data = JSON.parse(evt.data);
          if (data.type === 'telemetry' && data.payload) {
            const p = data.payload;
            setHasData(true);

            // Each CRAC reports only its own representative rack's power
            // (realistic ~10-28kW). Sum across every CRAC seen so far so
            // the facility-level KPI cards show a stable aggregate instead
            // of flickering between individual CRACs' small readings.
            if (p.crac_id && p.it_power_mw != null && p.cooling_power_mw != null) {
              cracPowerRef.current = {
                ...cracPowerRef.current,
                [p.crac_id]: { it_mw: p.it_power_mw, cooling_mw: p.cooling_power_mw },
              };
            }
            const agg = facilityAggregate(
              Object.values(cracPowerRef.current).map((r) => ({ it_power_mw: r.it_mw, cooling_power_mw: r.cooling_mw })),
              ZONE_SCALE,
            );

            setTelemetry((prev) => ({
              ...prev,
              server_inlet_temp_c: p.server_inlet_temp_c ?? prev.server_inlet_temp_c,
              server_outlet_temp_c: p.server_outlet_temp_c ?? prev.server_outlet_temp_c,
              fws_supply_temp_c: p.fws_supply_temp_c ?? prev.fws_supply_temp_c,
              return_temp_c: p.return_temp_c ?? prev.return_temp_c,
              flow_rate_lpm: p.flow_rate_lpm ?? prev.flow_rate_lpm,
              pump_speed_pct: p.pump_speed_pct ?? prev.pump_speed_pct,
              fan_speed_pct: p.fan_speed_pct ?? prev.fan_speed_pct,
              valve_split_pct: p.valve_split_pct ?? prev.valve_split_pct,
              it_power_kw: agg ? agg.itKw : prev.it_power_kw,
              cooling_power_kw: agg ? agg.coolingKw : prev.cooling_power_kw,
              pue: agg && agg.pue != null ? agg.pue : (p.pue ?? prev.pue),
              wue: p.wue ?? prev.wue,
              carbon_gco2_kwh: p.grid_carbon_gco2_kwh ?? prev.carbon_gco2_kwh,
              ashrae_status: p.ashrae_status ?? prev.ashrae_status,
            }));

            // Drive the 3D heatmap from live data: every rack served by this
            // CRAC takes the CRAC's live server-inlet temperature plus its
            // fixed positional offset (per-rack spatial variation within a
            // CRAC zone is illustrative; only the CRAC-level reading is live).
            if (p.crac_id && p.server_inlet_temp_c != null) {
              setSpatialGrid((prevGrid) => applyInletToGrid(prevGrid, p.crac_id, p.server_inlet_temp_c));
            }

            // SLA breach alarm: once when a unit enters a breach, again every 30 s while it lasts.
            const unitKey = p.crac_id || 'unit';
            const decision = alarmDecision(alarmStateRef.current[unitKey], p.ashrae_status, Date.now());
            alarmStateRef.current[unitKey] = decision.next;
            if (decision.raise) {
              setAlarms((prev) => [
                {
                  id: `${Date.now()}-${alarmIdCounter.current++}`,
                  timestamp: new Date().toLocaleTimeString(),
                  severity: p.ashrae_status,
                  message: `${p.crac_id || 'Unit'} zone: inlet ${Number(p.server_inlet_temp_c).toFixed(1)}°C outside the ASHRAE SLA envelope (18 to 27°C)`,
                },
                ...prev.slice(0, 9),
              ]);
            }
          }
        } catch {
          // Ignored parse error
        }
      };

      ws.onclose = () => {
        if (wsRef.current !== ws) return;
        setConnected(false);
        const delay = Math.min(10000, 1000 * Math.pow(1.5, reconnectAttempts.current));
        reconnectAttempts.current += 1;
        reconnectTimeoutRef.current = setTimeout(connectWs, delay);
      };

      ws.onerror = () => {
        if (wsRef.current === ws) ws.close();
      };
    } catch {
      setConnected(false);
    }
  }, [facilityId]);

  // Periodic REST poll fallback + synthetic simulation if offline
  //
  // `connected` is intentionally NOT a dependency here: it's set by this
  // same effect's WebSocket (via connectWs()'s onopen/onclose handlers), so
  // including it would re-run this effect every time the socket opens or
  // closes -- tearing down and recreating the connection in a destructive
  // loop (WS opens -> setConnected(true) -> effect re-fires -> cleanup
  // closes the socket it just opened -> reconnect -> repeat forever). The
  // interval reads the latest value via connectedRef instead.
  useEffect(() => {
    // Drop any accumulated per-CRAC power readings from the previous
    // facility so its stale numbers can't leak into this one's aggregate.
    cracPowerRef.current = {};
    alarmStateRef.current = {};
    setHasData(false);
    setAlarms([]);
    setSpatialGrid(generateInitialGrid());
    connectWs();

    pollIntervalRef.current = setInterval(async () => {
      if (!connectedRef.current) {
        try {
          const res = await fetch(`${API_BASE}/api/v1/telemetry/latest/${facilityId}`);
          if (res.ok) {
            const json = await res.json();
            setOffline(false);
            if (json.data && json.data.records && json.data.records.length > 0) {
              setHasData(true);
              const recs = json.data.records;
              const rec = recs[0];
              // Same facility aggregate as the WebSocket path: mean of the units x ZONE_SCALE.
              const agg = facilityAggregate(recs, ZONE_SCALE);
              setTelemetry((prev) => ({
                ...prev,
                ...rec,
                it_power_kw: agg ? agg.itKw : prev.it_power_kw,
                cooling_power_kw: agg ? agg.coolingKw : prev.cooling_power_kw,
                pue: agg && agg.pue != null ? agg.pue : (rec.pue ?? prev.pue),
              }));
              setSpatialGrid((prevGrid) => recs.reduce((g, r) => applyInletToGrid(g, r.crac_id, r.server_inlet_temp_c), prevGrid));
            }
          }
        } catch {
          setOffline(true);
          if (!demoMode) return;   // backend unreachable: keep the last real values, do not invent new ones
          // ?demo=1 only: synthetic micro-jitter so a standalone UI preview stays animated
          setTelemetry((prev) => {
            const jitter = (Math.random() - 0.5) * 0.1;
            const newInlet = Math.max(18.0, Math.min(26.5, prev.server_inlet_temp_c + jitter));
            return {
              ...prev,
              server_inlet_temp_c: Number(newInlet.toFixed(2)),
              pue: Number(Math.max(1.08, Math.min(1.22, prev.pue + (Math.random() - 0.5) * 0.002)).toFixed(3)),
              it_power_kw: Number((prev.it_power_kw + (Math.random() - 0.5) * 20.0).toFixed(1)),
            };
          });

          // Jitter 3D spatial grid nodes slightly
          setSpatialGrid((prevGrid) =>
            prevGrid.map((node) => {
              const delta = (Math.random() - 0.5) * 0.15;
              const nextTemp = Math.max(18.5, Math.min(27.8, node.temp_c + delta));
              return {
                ...node,
                temp_c: Number(nextTemp.toFixed(1)),
                ashrae_status: nextTemp > 27.0 ? 'SLA_BREACH' : 'NORMAL',
              };
            })
          );
        }
      }
    }, 1500);

    return () => {
      // Null the ref BEFORE closing: the socket's own onclose handler
      // checks `wsRef.current === ws` to decide whether to reconnect, and
      // that check needs to see this socket as already-superseded by the
      // time its (asynchronous) close event actually fires.
      const socket = wsRef.current;
      wsRef.current = null;
      if (socket) socket.close();
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, [connectWs, facilityId, demoMode]);

  // Submit operator control action
  const submitControlAction = async (cracId, actionPayload) => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/control/action/${facilityId}/${cracId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(actionPayload),
      });
      return await res.json();
    } catch (e) {
      if (demoMode) {   // ?demo=1 standalone preview only
        setTelemetry((prev) => ({ ...prev, ...actionPayload }));
        return { status: 'ok', local: true };
      }
      // Never report success for a command that did not reach the plant.
      return { status: 'error', detail: 'Backend unreachable: the command was NOT sent.' };
    }
  };

  // Toggle control mode (auto vs manual)
  const setControlMode = async (cracId, mode) => {
    try {
      const res = await fetch(`${API_BASE}/api/v1/control/mode/${facilityId}/${cracId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode }),
      });
      const data = await res.json();
      // Only reflect the new mode if the backend actually accepted it.
      if (res.ok) {
        setTelemetry((prev) => ({ ...prev, mode }));
      }
      return data;
    } catch {
      if (demoMode) {
        setTelemetry((prev) => ({ ...prev, mode }));
        return { status: 'ok', mode };
      }
      return { status: 'error', detail: 'Backend unreachable: the mode was NOT changed.' };
    }
  };

  return {
    connected,
    telemetry,
    spatialGrid,
    cracStatus,
    alarms,
    submitControlAction,
    setControlMode,
    offline,
    hasData,
  };
}
