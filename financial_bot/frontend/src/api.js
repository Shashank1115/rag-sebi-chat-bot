// src/api.js
export async function apiPOST(path, body = {}, opts = {}) {
  try {
    const res = await fetch(path, {
      method: 'POST',
      credentials: opts.credentials ?? 'include', // include by default (works across dev ports)
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        ...(opts.headers || {})
      },
      body: JSON.stringify(body)
    });

    const text = await res.text();
    let json = null;
    try { json = text ? JSON.parse(text) : {}; } catch (err) { json = null; }

    return { ok: res.ok, status: res.status, json, text, headers: res.headers };
  } catch (err) {
    console.error('apiPOST error', err);
    return { ok: false, status: 0, json: null, text: String(err) };
  }
}
