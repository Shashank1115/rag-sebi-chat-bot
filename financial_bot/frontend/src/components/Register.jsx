// src/components/Register.jsx
import React, { useState, useContext } from 'react';
import { apiPOST } from '../api';
import { AuthContext } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';

export default function Register() {
  const { setUser } = useContext(AuthContext);
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [err, setErr] = useState('');
  const [loading, setLoading] = useState(false);
  const nav = useNavigate();

  async function handleSubmit(e) {
    e.preventDefault();
    setErr('');
    if (!username.trim() || !password) { setErr('Provide username and password'); return; }
    if (password !== confirm) { setErr('Passwords do not match'); return; }
    setLoading(true);
    try {
      const { ok, json } = await apiPOST('/auth/register', { username: username.trim(), password, email });
      if (ok && json && json.success) {
        setUser(json.username || username);
        nav('/dashboard');
      } else {
        setErr((json && json.error) || `Register failed (${json && json.message ? json.message : 'server'})`);
      }
    } catch (e) {
      console.error(e);
      setErr('Network/server error');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-purple-100 p-6">
      <div className="max-w-md w-full bg-white rounded-2xl shadow-lg p-6">
        <h2 className="text-2xl font-semibold mb-4">Create an account</h2>
        {err && <div className="mb-3 text-sm text-red-600 bg-red-50 p-2 rounded">{err}</div>}
        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="Username"
            className="w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-300"
            autoComplete="username"
          />
          <input
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Email (optional)"
            className="w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-300"
            autoComplete="email"
          />
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password"
            className="w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-300"
            autoComplete="new-password"
          />
          <input
            type="password"
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
            placeholder="Confirm password"
            className="w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-300"
            autoComplete="new-password"
          />
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-green-600 text-white py-3 rounded-lg font-medium hover:bg-green-700 disabled:opacity-60"
          >
            {loading ? 'Creating...' : 'Create account'}
          </button>
        </form>

        <div className="mt-4 text-sm">
          <span>Already have an account? </span>
          <a className="text-blue-600 hover:underline" href="/login">Sign in</a>
        </div>
      </div>
    </div>
  );
}
