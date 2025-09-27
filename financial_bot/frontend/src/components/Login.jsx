// src/components/Login.jsx
import React, { useState, useContext } from 'react';
import { apiPOST } from '../api';
import { AuthContext } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';

export default function Login() {
  const { setUser } = useContext(AuthContext);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [err, setErr] = useState('');
  const [loading, setLoading] = useState(false);
  const nav = useNavigate();

  async function handleSubmit(e) {
    e.preventDefault();
    setErr('');
    if (!username.trim() || !password) { setErr('Username & password required'); return; }
    setLoading(true);
    try {
      const { ok, json, text, status } = await apiPOST('/auth/login', { username: username.trim(), password });

      if (ok && json && json.success) {
        setUser(json.username || json.user_id || username);
        nav('/dashboard');
        return;
      }

      const msg = (json && (json.error || json.message)) || text || `Login failed (status ${status})`;
      setErr(msg);
    } catch (e) {
      console.error('login error', e);
      setErr('Network/server error');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-purple-100 p-6">
      <div className="max-w-md w-full bg-white rounded-2xl shadow-lg p-6">
        <h2 className="text-2xl font-semibold mb-4">Sign in to SEBI Saathi</h2>
        {err && <div className="mb-3 text-sm text-red-600 bg-red-50 p-2 rounded">{err}</div>}
        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="Username"
            autoComplete="username"
            className="w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-300"
          />
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password"
            autoComplete="current-password"
            className="w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-300"
          />
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 text-white py-3 rounded-lg font-medium hover:bg-blue-700 disabled:opacity-60"
          >
            {loading ? 'Signing in...' : 'Sign in'}
          </button>
        </form>

        <div className="mt-4 text-sm">
          <span>Don't have an account? </span>
          <a className="text-blue-600 hover:underline" href="/register">Register</a>
        </div>
      </div>
    </div>
  );
}
