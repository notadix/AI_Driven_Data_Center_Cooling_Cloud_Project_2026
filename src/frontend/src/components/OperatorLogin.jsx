import React, { useState } from 'react';
import { LogIn, LogOut, User, X } from 'lucide-react';
import { API_BASE, COGNITO_CLIENT_ID, COGNITO_REGION } from '../config';

// Real Cognito sign-in, called directly from the browser against Cognito's public
// InitiateAuth endpoint (no SDK needed - it's a plain unauthenticated HTTPS/JSON call,
// the client ID is not a secret). On success the ID token is verified for real by the
// backend's /api/v1/auth/whoami (src/backend/api/v1/auth.py), which returns the
// operator's Cognito groups. This is additive: signing in or out never blocks any
// other part of the dashboard, and no other endpoint requires a token.

const COGNITO_IDP_URL = `https://cognito-idp.${COGNITO_REGION}.amazonaws.com/`;

async function cognitoSignIn(username, password) {
  const res = await fetch(COGNITO_IDP_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-amz-json-1.1',
      'X-Amz-Target': 'AWSCognitoIdentityProviderService.InitiateAuth',
    },
    body: JSON.stringify({
      AuthFlow: 'USER_PASSWORD_AUTH',
      ClientId: COGNITO_CLIENT_ID,
      AuthParameters: { USERNAME: username, PASSWORD: password },
    }),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(body.message || body.__type || 'Sign-in failed');
  }
  return body.AuthenticationResult?.IdToken;
}

async function fetchIdentity(idToken) {
  const res = await fetch(`${API_BASE}/api/v1/auth/whoami`, {
    headers: { Authorization: `Bearer ${idToken}` },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || 'Could not verify identity');
  }
  return res.json();
}

export default function OperatorLogin() {
  const [open, setOpen] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [identity, setIdentity] = useState(null); // { username, groups, sub } once signed in

  const handleSignIn = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError('');
    try {
      const idToken = await cognitoSignIn(username, password);
      const who = await fetchIdentity(idToken);
      setIdentity(who.identity);
      setOpen(false);
      setPassword('');
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const handleSignOut = () => {
    setIdentity(null);
    setUsername('');
  };

  if (identity) {
    const role = identity.groups?.[0] || 'Operator';
    return (
      <div className="flex items-center space-x-2 bg-slate-950/80 px-3 py-1.5 rounded-xl border border-slate-800 text-xs">
        <User className="w-3.5 h-3.5 text-cyan-400" />
        <span className="font-mono text-[11px] text-slate-300">{identity.username}</span>
        <span className="px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300 text-[10px] font-semibold">{role}</span>
        <button
          type="button"
          onClick={handleSignOut}
          title="Sign out"
          className="text-slate-500 hover:text-red-400 transition-colors"
        >
          <LogOut className="w-3.5 h-3.5" />
        </button>
      </div>
    );
  }

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex items-center space-x-1.5 bg-slate-950/80 px-3 py-1.5 rounded-xl border border-slate-800 text-xs text-slate-300 hover:border-cyan-500/40 hover:text-cyan-300 transition-colors"
      >
        <LogIn className="w-3.5 h-3.5" />
        <span className="font-mono text-[11px]">Sign In</span>
      </button>

      {open && (
        <form
          onSubmit={handleSignIn}
          className="absolute left-0 sm:left-auto sm:right-0 mt-2 w-64 max-w-[calc(100vw-2rem)] glass-panel rounded-xl border border-slate-800 p-4 z-50 space-y-2 shadow-2xl"
        >
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs font-semibold text-slate-300">Operator Sign In</span>
            <button type="button" onClick={() => setOpen(false)} className="text-slate-500 hover:text-slate-300">
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
          <input
            type="email"
            required
            placeholder="email"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-xs text-slate-200 outline-none focus:border-cyan-500/50"
          />
          <input
            type="password"
            required
            placeholder="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-xs text-slate-200 outline-none focus:border-cyan-500/50"
          />
          {error && <p className="text-[11px] text-red-400">{error}</p>}
          <button
            type="submit"
            disabled={busy}
            className="w-full bg-cyan-500/20 border border-cyan-500/40 text-cyan-300 rounded-lg py-1.5 text-xs font-semibold hover:bg-cyan-500/30 transition-colors disabled:opacity-50"
          >
            {busy ? 'Signing in...' : 'Sign In'}
          </button>
        </form>
      )}
    </div>
  );
}
