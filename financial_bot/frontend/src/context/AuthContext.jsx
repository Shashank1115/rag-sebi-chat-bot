// src/context/AuthContext.jsx
import React, { createContext, useState, useEffect } from 'react';

export const AuthContext = createContext({ user: null, setUser: () => {} });

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // Fetch current session on mount
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch('/auth/whoami', {
          credentials: 'include',
          headers: { Accept: 'application/json' }
        });
        if (res.ok) {
          const j = await res.json();
          if (j && j.username) setUser(j.username);
        }
      } catch (e) {
        console.warn("Auth check failed:", e);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  return (
    <AuthContext.Provider value={{ user, setUser, loading }}>
      {children}
    </AuthContext.Provider>
  );
}
