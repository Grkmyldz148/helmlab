#!/usr/bin/env node
/**
 * Helmlab Community Comments API
 * Standalone Node.js server — no dependencies
 * Stores comments in comments.json
 *
 * Usage: node server.js
 * Port: 3847 (or PORT env)
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const PORT = process.env.PORT || 3847;
const DATA_FILE = path.join(__dirname, 'comments.json');
const ADMIN_USER = process.env.ADMIN_USER || 'admin';
const ADMIN_PASS_HASH = process.env.ADMIN_PASS_HASH || '';
const TOKEN_SECRET = process.env.TOKEN_SECRET || crypto.randomBytes(32).toString('hex');

function sha256Sync(str) {
  return crypto.createHash('sha256').update(str).digest('hex');
}

// ── JSON file helpers ──────────────────────────────────
function readComments() {
  try {
    return JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));
  } catch {
    return [];
  }
}

function saveComments(comments) {
  fs.writeFileSync(DATA_FILE, JSON.stringify(comments, null, 2));
}

// ── Auth helpers ───────────────────────────────────────
function makeToken() {
  const ts = Date.now().toString();
  const sig = crypto.createHmac('sha256', TOKEN_SECRET).update(ts).digest('hex');
  return ts + '.' + sig;
}

function verifyToken(token) {
  if (!token) return false;
  const [ts, sig] = token.split('.');
  if (!ts || !sig) return false;
  if (Date.now() - parseInt(ts) > 86400000) return false; // 24h
  const expected = crypto.createHmac('sha256', TOKEN_SECRET).update(ts).digest('hex');
  return sig === expected;
}

function generateId() {
  return Date.now().toString(36) + crypto.randomBytes(3).toString('hex');
}

// ── HTTP helpers ───────────────────────────────────────
function json(res, data, status = 200) {
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, PATCH, DELETE, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  });
  res.end(JSON.stringify(data));
}

function parseBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', c => {
      body += c;
      if (body.length > 50000) reject(new Error('Too large'));
    });
    req.on('end', () => {
      try { resolve(JSON.parse(body)); }
      catch { reject(new Error('Invalid JSON')); }
    });
  });
}

// ── Server ─────────────────────────────────────────────
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  const p = url.pathname.replace(/\/+$/, '') || '/';
  const method = req.method;

  // CORS preflight
  if (method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PATCH, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    });
    return res.end();
  }

  try {
    // POST /auth
    if (p === '/auth' && method === 'POST') {
      const body = await parseBody(req);
      const passHash = sha256Sync(body.password || '');
      if (body.username === ADMIN_USER && passHash === ADMIN_PASS_HASH) {
        return json(res, { token: makeToken() });
      }
      return json(res, { error: 'Invalid credentials' }, 401);
    }

    // GET /comments — public (approved only)
    if (p === '/comments' && method === 'GET') {
      const status = url.searchParams.get('status') || 'approved';
      const comments = readComments()
        .filter(c => c.status === status)
        .map(({ id, name, message, created_at }) => ({ id, name, message, created_at }));
      return json(res, comments);
    }

    // POST /comments — submit new
    if (p === '/comments' && method === 'POST') {
      const body = await parseBody(req);
      if (!body.name || !body.message) return json(res, { error: 'Name and message required' }, 400);
      if (body.name.length > 100 || body.message.length > 2000) {
        return json(res, { error: 'Too long' }, 400);
      }
      const comments = readComments();
      comments.push({
        id: generateId(),
        name: body.name.trim(),
        email: (body.email || '').trim(),
        message: body.message.trim(),
        status: 'pending',
        created_at: new Date().toISOString(),
      });
      saveComments(comments);
      return json(res, { ok: true }, 201);
    }

    // ── Admin routes (auth required) ───────────────────
    const auth = (req.headers.authorization || '').replace('Bearer ', '');
    if (!verifyToken(auth)) return json(res, { error: 'Unauthorized' }, 401);

    // GET /admin/comments?status=pending
    if (p === '/admin/comments' && method === 'GET') {
      const status = url.searchParams.get('status') || 'pending';
      return json(res, readComments().filter(c => c.status === status));
    }

    // PATCH /admin/comments/:id
    const patchMatch = p.match(/^\/admin\/comments\/(.+)$/);
    if (patchMatch && method === 'PATCH') {
      const body = await parseBody(req);
      const comments = readComments();
      const idx = comments.findIndex(c => c.id === patchMatch[1]);
      if (idx === -1) return json(res, { error: 'Not found' }, 404);
      comments[idx].status = body.status;
      saveComments(comments);
      return json(res, { ok: true });
    }

    // DELETE /admin/comments/:id
    const delMatch = p.match(/^\/admin\/comments\/(.+)$/);
    if (delMatch && method === 'DELETE') {
      const comments = readComments().filter(c => c.id !== delMatch[1]);
      saveComments(comments);
      return json(res, { ok: true });
    }

    json(res, { error: 'Not found' }, 404);
  } catch (e) {
    json(res, { error: e.message }, 500);
  }
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`Helmlab API running on http://127.0.0.1:${PORT}`);
});
