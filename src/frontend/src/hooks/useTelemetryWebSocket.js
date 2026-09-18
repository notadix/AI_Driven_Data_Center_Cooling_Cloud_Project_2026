import { useState, useEffect, useRef, useCallback } from 'react';

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
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttempts = useRef(0);
  const pollIntervalRef = useRef(null);

  // Generate initial 8x8 matrix (64 racks: RACK-A01 to H08)
  function generateInitialGrid() {
    const rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];
    const grid = [];
    for (let r = 0; r < 8; r++) {
      for (let c = 1; c <= 8; c++) {
        const rowId = rows[r];
        const colId = c < 10 ? `0${c}` : `${c}`;
        const rackId = `RACK-${rowId}${colId}`;
        // Base nominal temp around 21 - 24°C with slight gradient
        const baseTemp = 21.0 + (r * 0.4) + (c * 0.2);
        grid.push({
          rack_id: rackId,
          row: r,
          col: c - 1,
          temp_c: Number(baseTemp.toFixed(1)),
          power_kw: Number((22.0 + Math.random() * 8.0).toFixed(1)),
          ashrae_status: baseTemp > 27.0 ? 'SLA_BREACH' : 'NORMAL',
          crac_id: r < 4 ? (c <= 4 ? 'CRAC-01' : 'CRAC-03') : (c <= 4 ? 'CRAC-02' : 'CRAC-04'),
        });
      }
    }
    return grid;
  }

  // Connect WebSocket
  const connectWs = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname || 'localhost';
    // Use port 8000 for backend during dev if running on 5173
    const port = window.location.port === '5173' ? '8000' : window.location.port;
    const wsUrl = `${protocol}//${host}:${port}/ws/stream?facility_id=${facilityId}`;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        reconnectAttempts.current = 0;
      };

      ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);
          if (data.type === 'telemetry' && data.payload) {
            const p = data.payload;
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
              // Backend payloads report power in MW (it_power_mw / cooling_power_mw);
              // this hook's state (and every consumer: Dashboard, PUEGauge, CarbonTracker)
              // uses kW, so convert on the way in instead of reading a key that never exists.
              it_power_kw: p.it_power_mw != null ? p.it_power_mw * 1000.0 : prev.it_power_kw,
              cooling_power_kw: p.cooling_power_mw != null ? p.cooling_power_mw * 1000.0 : prev.cooling_power_kw,
              pue: p.pue ?? prev.pue,
              ashrae_status: p.ashrae_status ?? prev.ashrae_status,
            }));

            // Check for SLA breach alarm
            if (p.ashrae_status === 'CRITICAL' || p.ashrae_status === 'SLA_BREACH') {
              setAlarms((prev) => [
                {
                  id: Date.now(),
                  timestamp: new Date().toLocaleTimeString(),
                  severity: p.ashrae_status,
                  message: `${p.rack_id || 'Rack'}: Inlet temp ${p.server_inlet_temp_c?.toFixed(1)}°C outside ASHRAE SLA envelope`,
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
        setConnected(false);
        const delay = Math.min(10000, 1000 * Math.pow(1.5, reconnectAttempts.current));
        reconnectAttempts.current += 1;
        reconnectTimeoutRef.current = setTimeout(connectWs, delay);
      };

      ws.onerror = () => {
        ws.close();
      };
    } catch {
      setConnected(false);
    }
  }, [facilityId]);

  // Periodic REST poll fallback + synthetic simulation if offline
  useEffect(() => {
    connectWs();

    pollIntervalRef.current = setInterval(async () => {
      if (!connected) {
        try {
          const res = await fetch(`http://localhost:8000/api/v1/telemetry/latest/${facilityId}`);
          if (res.ok) {
            const json = await res.json();
            if (json.data && json.data.records && json.data.records.length > 0) {
              const rec = json.data.records[0];
              setTelemetry((prev) => ({
                ...prev,
                ...rec,
                it_power_kw: rec.it_power_mw != null ? rec.it_power_mw * 1000.0 : prev.it_power_kw,
                cooling_power_kw: rec.cooling_power_mw != null ? rec.cooling_power_mw * 1000.0 : prev.cooling_power_kw,
              }));
            }
          }
        } catch {
          // If completely offline, run synthetic micro-jitter so UI remains dynamically alive
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
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, [connectWs, connected, facilityId]);

  // Submit operator control action
  const submitControlAction = async (cracId, actionPayload) => {
    try {
      const res = await fetch(`http://localhost:8000/api/v1/control/action/${cracId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(actionPayload),
      });
      return await res.json();
    } catch (e) {
      // Local optimistic update
      setTelemetry((prev) => ({ ...prev, ...actionPayload }));
      return { status: 'ok', local: true };
    }
  };

  // Toggle control mode (auto vs manual)
  const setControlMode = async (cracId, mode) => {
    try {
      const res = await fetch(`http://localhost:8000/api/v1/control/mode/${cracId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode }),
      });
      const data = await res.json();
      setTelemetry((prev) => ({ ...prev, mode }));
      return data;
    } catch {
      setTelemetry((prev) => ({ ...prev, mode }));
      return { status: 'ok', mode };
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
  };
}
