// Backend base URLs. The dashboard is served by Vite (dev: 5173, preview:
// 4173, or any other port) while the API always listens on 8000 unless
// overridden with VITE_API_BASE (e.g. "https://api.example.com").
const envBase = import.meta.env && import.meta.env.VITE_API_BASE;

const host = typeof window !== 'undefined' && window.location.hostname ? window.location.hostname : 'localhost';
const httpProto = typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'https:' : 'http:';
const wsProto = httpProto === 'https:' ? 'wss:' : 'ws:';

export const API_BASE = envBase ? envBase.replace(/\/$/, '') : `${httpProto}//${host}:8000`;
export const WS_BASE = envBase
  ? envBase.replace(/\/$/, '').replace(/^http/, 'ws')
  : `${wsProto}//${host}:8000`;

// Each simulated CRAC reports one representative rack (~10-28 kW); the backend scales it by this factor to
// the hall-scale load of the zone (src/digital_twin/physics_dynamics.py ZONE_SCALE). Keep in sync.
export const ZONE_SCALE = 1000;

// Cognito operator sign-in (components/OperatorLogin.jsx). This is additive: no route in the
// app requires a token, so a missing/expired login never blocks the dashboard. IDs match the
// real user pool created for this deployment (see docs/PROJECT_EXPLAINED.md section 3).
export const COGNITO_REGION = import.meta.env.VITE_COGNITO_REGION || 'us-east-1';
export const COGNITO_CLIENT_ID = import.meta.env.VITE_COGNITO_CLIENT_ID || '34mj0l4b9stj3ecr2lkmoe14kk';
