#!/usr/bin/env node
/**
 * Helmlab API — Comments + Blog + Image Upload
 * MongoDB Atlas backend, token-based admin auth
 *
 * Collections: comments, blogs
 * Port: 3847 (or PORT env)
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const PORT = process.env.PORT || 3847;
const ADMIN_USER = process.env.ADMIN_USER || 'admin';
const ADMIN_PASS_HASH = process.env.ADMIN_PASS_HASH || '';
const TOKEN_SECRET = process.env.TOKEN_SECRET || crypto.randomBytes(32).toString('hex');
const MONGO_URI = process.env.MONGO_URI || 'REDACTED_MONGO_URI';
const UPLOAD_DIR = process.env.UPLOAD_DIR || '/home/ismailyagci/web/helmlab.space/public_html/uploads/blog';
const BLOG_SECRET = process.env.BLOG_SECRET || 'REDACTED_BLOG_SECRET';

// ── MongoDB ───────────────────────────────────────────────
const { MongoClient, ObjectId } = require('mongodb');
let db;

async function connectDB() {
  const client = new MongoClient(MONGO_URI);
  await client.connect();
  db = client.db('helmlab');
  console.log('MongoDB connected');

  // Ensure indexes
  await db.collection('blogs').createIndex({ slug: 1 }, { unique: true });
  await db.collection('blogs').createIndex({ status: 1, created_at: -1 });
  await db.collection('comments').createIndex({ status: 1, created_at: -1 });

  // Migrate comments.json if exists and collection is empty
  const count = await db.collection('comments').countDocuments();
  const jsonPath = path.join(__dirname, 'comments.json');
  if (count === 0 && fs.existsSync(jsonPath)) {
    try {
      const old = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
      if (old.length > 0) {
        await db.collection('comments').insertMany(old);
        console.log(`Migrated ${old.length} comments from JSON to MongoDB`);
      }
    } catch (e) {
      console.error('Migration failed:', e.message);
    }
  }
}

// ── Auth helpers ──────────────────────────────────────────
function sha256(str) {
  return crypto.createHash('sha256').update(str).digest('hex');
}

function makeToken() {
  const ts = Date.now().toString();
  const sig = crypto.createHmac('sha256', TOKEN_SECRET).update(ts).digest('hex');
  return ts + '.' + sig;
}

function verifyToken(token) {
  if (!token) return false;
  const [ts, sig] = token.split('.');
  if (!ts || !sig) return false;
  if (Date.now() - parseInt(ts) > 86400000) return false;
  const expected = crypto.createHmac('sha256', TOKEN_SECRET).update(ts).digest('hex');
  return sig === expected;
}

// ── Slug generator ────────────────────────────────────────
function slugify(text) {
  return text
    .toString().toLowerCase().trim()
    .replace(/[çÇ]/g, 'c').replace(/[ğĞ]/g, 'g')
    .replace(/[ıİ]/g, 'i').replace(/[öÖ]/g, 'o')
    .replace(/[şŞ]/g, 's').replace(/[üÜ]/g, 'u')
    .replace(/[^\w\s-]/g, '').replace(/[\s_]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

// ── HTTP helpers ──────────────────────────────────────────
function json(res, data, status = 200) {
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, PUT, PATCH, DELETE, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Blog-Secret',
  });
  res.end(JSON.stringify(data));
}

function parseBody(req, limit = 50000) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', c => {
      body += c;
      if (body.length > limit) reject(new Error('Too large'));
    });
    req.on('end', () => {
      try { resolve(JSON.parse(body)); }
      catch { reject(new Error('Invalid JSON')); }
    });
  });
}

// Parse multipart/form-data (minimal, for image upload)
function parseMultipart(req) {
  return new Promise((resolve, reject) => {
    const contentType = req.headers['content-type'] || '';
    const boundaryMatch = contentType.match(/boundary=(.+)/);
    if (!boundaryMatch) return reject(new Error('No boundary'));
    const boundary = '--' + boundaryMatch[1];

    const chunks = [];
    let totalSize = 0;
    const MAX_SIZE = 10 * 1024 * 1024; // 10MB

    req.on('data', chunk => {
      totalSize += chunk.length;
      if (totalSize > MAX_SIZE) {
        req.destroy();
        return reject(new Error('File too large (max 10MB)'));
      }
      chunks.push(chunk);
    });

    req.on('end', () => {
      const buf = Buffer.concat(chunks);
      const parts = {};
      const str = buf.toString('binary');
      const sections = str.split(boundary).slice(1, -1);

      for (const section of sections) {
        const headerEnd = section.indexOf('\r\n\r\n');
        if (headerEnd === -1) continue;
        const header = section.substring(0, headerEnd);
        const body = section.substring(headerEnd + 4, section.length - 2);

        const nameMatch = header.match(/name="([^"]+)"/);
        if (!nameMatch) continue;
        const name = nameMatch[1];

        const filenameMatch = header.match(/filename="([^"]+)"/);
        if (filenameMatch) {
          const contentTypeMatch = header.match(/Content-Type:\s*(.+)/i);
          parts[name] = {
            filename: filenameMatch[1],
            contentType: contentTypeMatch ? contentTypeMatch[1].trim() : 'application/octet-stream',
            data: Buffer.from(body, 'binary'),
          };
        } else {
          parts[name] = body.trim();
        }
      }
      resolve(parts);
    });

    req.on('error', reject);
  });
}

// ── Image upload handler ──────────────────────────────────
async function handleImageUpload(req) {
  const parts = await parseMultipart(req);
  const file = parts.image || parts.file;
  if (!file || !file.data) throw new Error('No image file');

  const allowed = ['image/jpeg', 'image/png', 'image/webp', 'image/gif', 'image/svg+xml'];
  if (!allowed.includes(file.contentType)) throw new Error('Invalid image type');

  const ext = {
    'image/jpeg': '.jpg', 'image/png': '.png', 'image/webp': '.webp',
    'image/gif': '.gif', 'image/svg+xml': '.svg',
  }[file.contentType] || '.bin';

  const name = Date.now().toString(36) + crypto.randomBytes(4).toString('hex') + ext;
  fs.mkdirSync(UPLOAD_DIR, { recursive: true });
  fs.writeFileSync(path.join(UPLOAD_DIR, name), file.data);

  return { url: '/uploads/blog/' + name, filename: name, size: file.data.length };
}

// ── Sitemap generator ─────────────────────────────────────
async function generateSitemap() {
  const blogs = await db.collection('blogs')
    .find({ status: 'published' })
    .sort({ created_at: -1 })
    .project({ slug: 1, updated_at: 1, created_at: 1 })
    .toArray();

  const now = new Date().toISOString().split('T')[0];
  let xml = '<?xml version="1.0" encoding="UTF-8"?>\n';
  xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n';

  // Static pages
  const pages = [
    { loc: 'https://helmlab.space/', priority: '1.0', changefreq: 'weekly' },
    { loc: 'https://helmlab.space/palette.html', priority: '0.8', changefreq: 'monthly' },
    { loc: 'https://helmlab.space/blog.html', priority: '0.8', changefreq: 'weekly' },
  ];
  for (const p of pages) {
    xml += `  <url>\n    <loc>${p.loc}</loc>\n    <lastmod>${now}</lastmod>\n    <changefreq>${p.changefreq}</changefreq>\n    <priority>${p.priority}</priority>\n  </url>\n`;
  }

  // Blog posts
  for (const b of blogs) {
    const date = (b.updated_at || b.created_at).toISOString().split('T')[0];
    xml += `  <url>\n    <loc>https://helmlab.space/blog.html?slug=${b.slug}</loc>\n    <lastmod>${date}</lastmod>\n    <changefreq>monthly</changefreq>\n    <priority>0.6</priority>\n  </url>\n`;
  }

  xml += '</urlset>';
  return xml;
}

// ── Server ────────────────────────────────────────────────
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  const p = url.pathname.replace(/\/+$/, '') || '/';
  const method = req.method;

  if (method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, PATCH, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Blog-Secret',
    });
    return res.end();
  }

  try {
    // ── Auth ────────────────────────────────────────────
    if (p === '/auth' && method === 'POST') {
      const body = await parseBody(req);
      if (body.username === ADMIN_USER && sha256(body.password || '') === ADMIN_PASS_HASH) {
        return json(res, { token: makeToken() });
      }
      return json(res, { error: 'Invalid credentials' }, 401);
    }

    // ── Public: Comments ────────────────────────────────
    if (p === '/comments' && method === 'GET') {
      const comments = await db.collection('comments')
        .find({ status: 'approved' })
        .sort({ created_at: -1 })
        .project({ _id: 0, id: 1, name: 1, message: 1, created_at: 1 })
        .toArray();
      return json(res, comments);
    }

    if (p === '/comments' && method === 'POST') {
      const body = await parseBody(req);
      if (!body.name || !body.message) return json(res, { error: 'Name and message required' }, 400);
      if (body.name.length > 100 || body.message.length > 2000) return json(res, { error: 'Too long' }, 400);
      await db.collection('comments').insertOne({
        id: Date.now().toString(36) + crypto.randomBytes(3).toString('hex'),
        name: body.name.trim(),
        email: (body.email || '').trim(),
        message: body.message.trim(),
        status: 'pending',
        created_at: new Date(),
      });
      return json(res, { ok: true }, 201);
    }

    // ── Public: Blogs ───────────────────────────────────
    if (p === '/blogs' && method === 'GET') {
      const page = parseInt(url.searchParams.get('page')) || 1;
      const limit = Math.min(parseInt(url.searchParams.get('limit')) || 10, 50);
      const tag = url.searchParams.get('tag');

      const query = { status: 'published' };
      if (tag) query.tags = tag;

      const [blogs, total] = await Promise.all([
        db.collection('blogs')
          .find(query)
          .sort({ created_at: -1 })
          .skip((page - 1) * limit)
          .limit(limit)
          .project({ content: 0 })
          .toArray(),
        db.collection('blogs').countDocuments(query),
      ]);

      return json(res, { blogs, total, page, pages: Math.ceil(total / limit) });
    }

    // GET /blogs/:slug
    const blogSlugMatch = p.match(/^\/blogs\/(.+)$/);
    if (blogSlugMatch && method === 'GET') {
      const blog = await db.collection('blogs').findOne({
        slug: blogSlugMatch[1],
        status: 'published',
      });
      if (!blog) return json(res, { error: 'Not found' }, 404);
      return json(res, blog);
    }

    // ── Public: Sitemap ─────────────────────────────────
    if (p === '/sitemap.xml' && method === 'GET') {
      const xml = await generateSitemap();
      res.writeHead(200, { 'Content-Type': 'application/xml', 'Access-Control-Allow-Origin': '*' });
      return res.end(xml);
    }

    // ── CI/CD: Auto blog post ───────────────────────────
    if (p === '/ci/blog' && method === 'POST') {
      const secret = req.headers['x-blog-secret'];
      if (secret !== BLOG_SECRET) return json(res, { error: 'Forbidden' }, 403);

      const body = await parseBody(req, 500000);
      const slug = slugify(body.title);
      const existing = await db.collection('blogs').findOne({ slug });
      if (existing) {
        await db.collection('blogs').updateOne({ slug }, {
          $set: { content: body.content, excerpt: body.excerpt, updated_at: new Date() },
        });
        return json(res, { ok: true, action: 'updated', slug });
      }

      await db.collection('blogs').insertOne({
        title: body.title,
        slug,
        content: body.content,
        excerpt: body.excerpt || '',
        cover_image: body.cover_image || '',
        tags: body.tags || ['release'],
        author: body.author || 'Helmlab CI',
        status: 'published',
        created_at: new Date(),
        updated_at: new Date(),
      });
      return json(res, { ok: true, action: 'created', slug });
    }

    // ── Admin routes (auth required) ────────────────────
    const auth = (req.headers.authorization || '').replace('Bearer ', '');
    if (!verifyToken(auth)) return json(res, { error: 'Unauthorized' }, 401);

    // -- Admin: Comments --
    if (p === '/admin/comments' && method === 'GET') {
      const status = url.searchParams.get('status') || 'pending';
      const comments = await db.collection('comments')
        .find({ status })
        .sort({ created_at: -1 })
        .toArray();
      return json(res, comments);
    }

    const commentPatch = p.match(/^\/admin\/comments\/(.+)$/);
    if (commentPatch && method === 'PATCH') {
      const body = await parseBody(req);
      await db.collection('comments').updateOne(
        { id: commentPatch[1] },
        { $set: { status: body.status } },
      );
      return json(res, { ok: true });
    }

    if (commentPatch && method === 'DELETE') {
      await db.collection('comments').deleteOne({ id: commentPatch[1] });
      return json(res, { ok: true });
    }

    // -- Admin: Blogs --
    if (p === '/admin/blogs' && method === 'GET') {
      const status = url.searchParams.get('status');
      const query = status ? { status } : {};
      const blogs = await db.collection('blogs')
        .find(query)
        .sort({ created_at: -1 })
        .project({ content: 0 })
        .toArray();
      return json(res, blogs);
    }

    if (p === '/admin/blogs' && method === 'POST') {
      const body = await parseBody(req, 500000);
      if (!body.title || !body.content) return json(res, { error: 'Title and content required' }, 400);
      const slug = slugify(body.title);

      const existing = await db.collection('blogs').findOne({ slug });
      if (existing) return json(res, { error: 'Slug already exists' }, 409);

      const result = await db.collection('blogs').insertOne({
        title: body.title,
        slug,
        content: body.content,
        excerpt: body.excerpt || '',
        cover_image: body.cover_image || '',
        tags: body.tags || [],
        author: body.author || 'Admin',
        status: body.status || 'draft',
        created_at: new Date(),
        updated_at: new Date(),
      });
      return json(res, { ok: true, id: result.insertedId, slug }, 201);
    }

    const blogAdminMatch = p.match(/^\/admin\/blogs\/(.+)$/);
    if (blogAdminMatch && method === 'GET') {
      const blog = await db.collection('blogs').findOne({ _id: new ObjectId(blogAdminMatch[1]) });
      if (!blog) return json(res, { error: 'Not found' }, 404);
      return json(res, blog);
    }

    if (blogAdminMatch && method === 'PUT') {
      const body = await parseBody(req, 500000);
      const update = { updated_at: new Date() };
      if (body.title) { update.title = body.title; update.slug = slugify(body.title); }
      if (body.content !== undefined) update.content = body.content;
      if (body.excerpt !== undefined) update.excerpt = body.excerpt;
      if (body.cover_image !== undefined) update.cover_image = body.cover_image;
      if (body.tags) update.tags = body.tags;
      if (body.status) update.status = body.status;
      if (body.author) update.author = body.author;

      await db.collection('blogs').updateOne(
        { _id: new ObjectId(blogAdminMatch[1]) },
        { $set: update },
      );
      return json(res, { ok: true });
    }

    if (blogAdminMatch && method === 'DELETE') {
      await db.collection('blogs').deleteOne({ _id: new ObjectId(blogAdminMatch[1]) });
      return json(res, { ok: true });
    }

    // -- Admin: Image upload --
    if (p === '/admin/upload' && method === 'POST') {
      const result = await handleImageUpload(req);
      return json(res, result, 201);
    }

    json(res, { error: 'Not found' }, 404);
  } catch (e) {
    console.error('API Error:', e.message);
    json(res, { error: e.message }, 500);
  }
});

// ── Start ─────────────────────────────────────────────────
connectDB().then(() => {
  server.listen(PORT, '127.0.0.1', () => {
    console.log(`Helmlab API running on http://127.0.0.1:${PORT}`);
  });
}).catch(err => {
  console.error('Failed to connect MongoDB:', err.message);
  process.exit(1);
});
