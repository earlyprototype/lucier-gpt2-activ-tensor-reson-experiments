#!/usr/bin/env node
/**
 * Headless browser smoke + interaction test for docs/graph/viewer.html.
 *
 * Serves docs/graph on a free port, drives the viewer with Playwright's
 * bundled Chromium, and asserts eight things:
 *
 *   1. viewer.html loads with ZERO pageerror and ZERO console-error events
 *   2. a <canvas> exists and vis-network has actually rendered nodes
 *   3. Dissolution graph loads dissolution.json + re-renders on model switch
 *   4. Isomorphism graph renders
 *   5. search surfaces a known id; clicking that node opens details + status pill
 *   6. the timeline scrubber hides nodes at min and restores them at max
 *   7. "copy evidence chain" produces a non-empty chain string
 *   8. no HTTP >= 400 responses for any requested asset
 *
 * Run:  node docs/graph/tests/smoke_test.mjs
 * Exit: 0 = all pass, 1 = at least one failure.
 *
 * The two CDN <script> tags (vis-network, marked) are served from a local
 * mirror via request interception so the suite is hermetic and does not
 * silently pass/fail on CDN weather. The mirror is populated once with curl.
 */

import http from 'node:http';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);

// Resolve Playwright from wherever it actually lives: a local/parent
// node_modules first, then PLAYWRIGHT_PATH, then the global install used by
// this repo's container. Hardcoding one absolute path makes the suite
// unrunnable for anyone who clones the repo.
function loadChromium() {
    const candidates = [
        'playwright',
        process.env.PLAYWRIGHT_PATH,
        '/opt/node22/lib/node_modules/playwright',
        '/usr/lib/node_modules/playwright',
        '/usr/local/lib/node_modules/playwright'
    ].filter(Boolean);
    const tried = [];
    for (const c of candidates) {
        try {
            return require(c).chromium;
        } catch (err) {
            tried.push(`${c} (${err.code || err.message})`);
        }
    }
    console.error(
        'Could not load Playwright. Install it with `npm i -D playwright` ' +
        'or set PLAYWRIGHT_PATH to an existing install.\nTried:\n  ' +
        tried.join('\n  ')
    );
    process.exit(1);
}

const chromium = loadChromium();

const HERE = path.dirname(fileURLToPath(import.meta.url));
// GRAPH_ROOT exists so the suite can be pointed at a fault-injected copy of the
// directory as a negative control (proving the collectors can actually fail).
const ROOT = path.resolve(process.env.GRAPH_ROOT || path.resolve(HERE, '..'));

// --- CDN mirror ------------------------------------------------------------
const CDN_ASSETS = [
    { url: 'https://unpkg.com/vis-network/standalone/umd/vis-network.min.js', file: 'vis-network.min.js' },
    { url: 'https://cdn.jsdelivr.net/npm/marked/marked.min.js', file: 'marked.min.js' }
];
const MIRROR_DIR = path.join(process.env.GRAPH_TEST_CACHE || path.join(os.tmpdir(), 'graph-smoke-cache'), 'vendor');

function ensureMirror() {
    fs.mkdirSync(MIRROR_DIR, { recursive: true });
    for (const asset of CDN_ASSETS) {
        const dest = path.join(MIRROR_DIR, asset.file);
        if (fs.existsSync(dest) && fs.statSync(dest).size > 1000) continue;
        process.stdout.write(`  fetching mirror copy of ${asset.file} ...\n`);
        execFileSync('curl', ['-sSL', '--max-time', '90', '-o', dest, asset.url], { stdio: 'inherit' });
        if (!fs.existsSync(dest) || fs.statSync(dest).size < 1000) {
            throw new Error(`could not mirror ${asset.url}`);
        }
    }
}

// --- static server ---------------------------------------------------------
const MIME = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.mjs': 'text/javascript; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.md': 'text/markdown; charset=utf-8',
    '.svg': 'image/svg+xml',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.ico': 'image/x-icon'
};

function startServer(rootDir) {
    return new Promise((resolve, reject) => {
        const server = http.createServer(async (req, res) => {
            let rel;
            try {
                rel = decodeURIComponent(new URL(req.url, 'http://x').pathname);
            } catch {
                res.writeHead(400).end('bad url');
                return;
            }
            if (rel.endsWith('/')) rel += 'index.html';
            const abs = path.join(rootDir, rel);
            // never serve outside the root
            if (!abs.startsWith(rootDir)) {
                res.writeHead(403).end('forbidden');
                return;
            }
            try {
                const stat = await fsp.stat(abs);
                if (stat.isDirectory()) throw new Error('dir');
                const body = await fsp.readFile(abs);
                res.writeHead(200, {
                    'content-type': MIME[path.extname(abs).toLowerCase()] || 'application/octet-stream',
                    'content-length': body.length,
                    'cache-control': 'no-store'
                });
                res.end(body);
            } catch {
                res.writeHead(404, { 'content-type': 'text/plain' }).end('not found');
            }
        });
        server.on('error', reject);
        server.listen(0, '127.0.0.1', () => resolve(server));
    });
}

// --- reporting -------------------------------------------------------------
const results = [];
function record(n, title, ok, detail) {
    results.push({ n, title, ok, detail });
    const tag = ok ? '\x1b[32mPASS\x1b[0m' : '\x1b[31mFAIL\x1b[0m';
    console.log(`${tag}  ${n}. ${title}`);
    if (detail) {
        for (const line of String(detail).split('\n')) console.log(`        ${line}`);
    }
}

// --- error / response collectors -------------------------------------------
const pageErrors = [];   // {phase, text}
const consoleErrors = [];// {phase, text}
const badResponses = []; // {phase, status, url}
const allResponses = [];
let phase = 'boot';

function snapshotErrors() {
    return { pe: pageErrors.length, ce: consoleErrors.length, bad: badResponses.length };
}
function errorsSince(snap) {
    return {
        pageErrors: pageErrors.slice(snap.pe),
        consoleErrors: consoleErrors.slice(snap.ce),
        badResponses: badResponses.slice(snap.bad)
    };
}
function describeSince(snap) {
    const d = errorsSince(snap);
    const out = [];
    d.pageErrors.forEach(e => out.push(`pageerror[${e.phase}]: ${e.text}`));
    d.consoleErrors.forEach(e => out.push(`console.error[${e.phase}]: ${e.text}`));
    d.badResponses.forEach(e => out.push(`http ${e.status}[${e.phase}]: ${e.url}`));
    return out;
}
function cleanSince(snap) {
    const d = errorsSince(snap);
    return d.pageErrors.length === 0 && d.consoleErrors.length === 0 && d.badResponses.length === 0;
}

// --- page helpers ----------------------------------------------------------
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

async function waitLoaded(page) {
    // The viewer hides #loading once vis has stabilised (or after its own 10s
    // safety timeout). Wait for the overlay to go away, then let physics rest.
    await page.waitForFunction(() => {
        const el = document.getElementById('loading');
        return el && (el.classList.contains('hidden') || getComputedStyle(el).display === 'none');
    }, null, { timeout: 30000 });
    await settle(page);
}

// Poll node positions until they stop moving, so real mouse clicks land.
async function settle(page, tries = 30) {
    let prev = null;
    for (let i = 0; i < tries; i++) {
        const sig = await page.evaluate(() => {
            if (typeof network === 'undefined' || !network) return null;
            const pos = network.getPositions();
            const ids = Object.keys(pos).sort().slice(0, 25);
            return ids.map(id => `${id}:${Math.round(pos[id].x)},${Math.round(pos[id].y)}`).join('|');
        });
        if (sig !== null && sig === prev) return true;
        prev = sig;
        await sleep(250);
    }
    return false;
}

// Everything the assertions need to know about the current render, read
// straight out of the live vis DataSet.
async function readState(page) {
    return page.evaluate(() => {
        const has = typeof network !== 'undefined' && network;
        const ds = typeof nodesDataSet !== 'undefined' ? nodesDataSet : null;
        const es = typeof edgesDataSet !== 'undefined' ? edgesDataSet : null;
        const items = ds ? ds.get() : [];
        return {
            hasNetwork: !!has,
            datasetLength: ds ? ds.length : 0,
            edgeLength: es ? es.length : 0,
            visibleNodes: items.filter(n => n.hidden !== true).length,
            visibleIds: items.filter(n => n.hidden !== true).map(n => n.id),
            renderMode: typeof renderMode !== 'undefined' ? renderMode : null,
            graphKey: typeof currentGraphKey !== 'undefined' ? currentGraphKey : null,
            modelKey: typeof currentModelKey !== 'undefined' ? currentModelKey : null,
            title: (document.getElementById('graph-title') || {}).textContent || '',
            canvasCount: document.querySelectorAll('#graph canvas').length
        };
    });
}

// Non-blank check: sample the vis canvas and count pixels that differ from the
// dominant (background) colour.
async function canvasInk(page) {
    return page.evaluate(() => {
        const canvas = document.querySelector('#graph canvas');
        if (!canvas) return { ok: false, reason: 'no canvas element' };
        const w = canvas.width, h = canvas.height;
        if (!w || !h) return { ok: false, reason: `canvas has zero size (${w}x${h})` };
        const ctx = canvas.getContext('2d');
        if (!ctx) return { ok: false, reason: 'no 2d context' };
        const data = ctx.getImageData(0, 0, w, h).data;
        const counts = new Map();
        let total = 0;
        for (let i = 0; i < data.length; i += 4 * 7) { // stride-sample
            const key = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
            counts.set(key, (counts.get(key) || 0) + 1);
            total++;
        }
        let bg = 0, bgCount = 0;
        for (const [k, v] of counts) if (v > bgCount) { bgCount = v; bg = k; }
        const nonBg = total - bgCount;
        return {
            ok: true, w, h, total, nonBg,
            distinctColours: counts.size,
            fraction: total ? nonBg / total : 0
        };
    });
}

async function typeSearch(page, term) {
    await page.fill('#search', '');
    await page.fill('#search', term);
    await page.waitForTimeout(250);
}

// Convert a vis node's canvas position into viewport coordinates and click it
// for real (no synthetic showNodeDetails call).
async function clickNode(page, nodeId) {
    await page.evaluate((id) => {
        network.moveTo({ position: network.getPositions([id])[id], scale: 1.2, animation: false });
    }, nodeId);
    await page.waitForTimeout(200);
    const pt = await page.evaluate((id) => {
        const pos = network.getPositions([id])[id];
        const dom = network.canvasToDOM(pos);
        const rect = document.getElementById('graph').getBoundingClientRect();
        return { x: rect.left + dom.x, y: rect.top + dom.y };
    }, nodeId);
    await page.mouse.click(pt.x, pt.y);
    await page.waitForTimeout(350);
    return pt;
}

// --- main ------------------------------------------------------------------
async function main() {
    console.log('\n=== viewer.html browser smoke + interaction test ===\n');
    ensureMirror();

    const server = await startServer(ROOT);
    const port = server.address().port;
    const base = `http://127.0.0.1:${port}`;
    console.log(`  serving ${ROOT} at ${base}`);
    console.log(`  vis-network + marked served from local mirror ${MIRROR_DIR} ` +
        `(the page still requests the CDN URLs; interception keeps the run hermetic)\n`);

    const mirror = Object.fromEntries(
        CDN_ASSETS.map(a => [a.url, fs.readFileSync(path.join(MIRROR_DIR, a.file), 'utf8')])
    );

    const browser = await chromium.launch({ headless: true });
    const context = await browser.newContext({ viewport: { width: 1600, height: 1000 } });
    const page = await context.newPage();

    // Serve the two CDN scripts from the local mirror; anything else that tries
    // to leave the box is a finding, so let it through and let the response
    // recorder see the result.
    await page.route('**/*', async (route) => {
        const url = route.request().url();
        for (const asset of CDN_ASSETS) {
            if (url === asset.url || url.startsWith(asset.url.split('?')[0])) {
                await route.fulfill({
                    status: 200,
                    contentType: 'text/javascript; charset=utf-8',
                    body: mirror[asset.url]
                });
                return;
            }
        }
        await route.continue();
    });

    page.on('pageerror', (err) => {
        pageErrors.push({ phase, text: (err && err.stack) || String(err) });
    });
    page.on('console', (msg) => {
        if (msg.type() !== 'error') return;
        consoleErrors.push({ phase, text: msg.text() });
    });
    page.on('requestfailed', (req) => {
        consoleErrors.push({ phase, text: `requestfailed ${req.url()} — ${(req.failure() || {}).errorText}` });
    });
    page.on('response', (res) => {
        const rec = { phase, status: res.status(), url: res.url() };
        allResponses.push(rec);
        if (rec.status >= 400) badResponses.push(rec);
    });

    let exitCode = 0;
    try {
        // ---------------------------------------------------------------- 1
        phase = 'initial-load';
        const snap1 = snapshotErrors();
        await page.goto(`${base}/viewer.html`, { waitUntil: 'domcontentloaded' });
        await waitLoaded(page);
        const s1 = await readState(page);
        record(1, 'viewer.html loads with zero pageerror and zero console-error events',
            pageErrors.length === 0 && consoleErrors.length === 0,
            pageErrors.length || consoleErrors.length
                ? describeSince(snap1).join('\n')
                : `graph="${s1.graphKey}" mode="${s1.renderMode}" title="${s1.title}"`);

        // ---------------------------------------------------------------- 2
        phase = 'render-check';
        const ink = await canvasInk(page);
        const ok2 = s1.canvasCount > 0 && s1.hasNetwork && s1.datasetLength > 0 &&
            s1.visibleNodes > 0 && ink.ok && ink.fraction > 0.005;
        record(2, 'a <canvas> is present and vis-network rendered nodes (non-blank pixels)', ok2,
            `canvases=${s1.canvasCount} network=${s1.hasNetwork} DataSet.length=${s1.datasetLength} ` +
            `visible=${s1.visibleNodes} edges=${s1.edgeLength}\n` +
            (ink.ok
                ? `canvas ${ink.w}x${ink.h}, ${ink.distinctColours} distinct colours, ` +
                  `${(ink.fraction * 100).toFixed(1)}% non-background pixels`
                : `canvas unreadable: ${ink.reason}`));

        const evidenceCount = s1.datasetLength;

        // ---------------------------------------------------------------- 3
        phase = 'dissolution';
        const snap3 = snapshotErrors();
        const respPromise = page.waitForResponse(
            r => r.url().includes('_data/dissolution.json'), { timeout: 20000 }
        ).catch(() => null);
        await page.selectOption('#graph-select', 'dissolution');
        const dissolutionResp = await respPromise;
        await waitLoaded(page);
        const s3 = await readState(page);
        const ink3 = await canvasInk(page);
        const dissolutionOk = !!dissolutionResp && dissolutionResp.status() === 200 &&
            s3.renderMode === 'dissolution' && s3.datasetLength > 0 && s3.visibleNodes > 0 &&
            ink3.ok && ink3.fraction > 0.005 && cleanSince(snap3);

        // model selector re-render
        phase = 'dissolution-model-switch';
        const snap3b = snapshotErrors();
        const models = await page.$$eval('#model-select option', os => os.map(o => o.value));
        const firstModel = s3.modelKey;
        const otherModel = models.find(m => m !== firstModel);
        let s3b = null, modelOk = false, modelDetail = 'no second model available';
        if (otherModel) {
            await page.selectOption('#model-select', otherModel);
            await page.waitForFunction(
                (k) => typeof currentModelKey !== 'undefined' && currentModelKey === k,
                otherModel, { timeout: 20000 });
            await waitLoaded(page);
            s3b = await readState(page);
            const ink3b = await canvasInk(page);
            modelOk = s3b.modelKey === otherModel && s3b.datasetLength > 0 && s3b.visibleNodes > 0 &&
                ink3b.ok && ink3b.fraction > 0.005 && cleanSince(snap3b);
            modelDetail = `${firstModel} (${s3.datasetLength} nodes) -> ${otherModel} ` +
                `(${s3b.datasetLength} nodes, ${s3b.edgeLength} edges), ` +
                `${(ink3b.fraction * 100).toFixed(1)}% non-bg pixels`;
        }
        record(3, 'Dissolution graph loads dissolution.json + renders; model selector re-renders',
            dissolutionOk && modelOk,
            `dissolution.json -> HTTP ${dissolutionResp ? dissolutionResp.status() : 'NOT REQUESTED'}; ` +
            `mode=${s3.renderMode}, ${s3.datasetLength} nodes, ${s3.edgeLength} edges\n` +
            `model switch: ${modelDetail}\n` +
            (describeSince(snap3).join('\n') || 'no errors during either step'));

        // ---------------------------------------------------------------- 4
        phase = 'isomorphism';
        const snap4 = snapshotErrors();
        const isoPromise = page.waitForResponse(
            r => r.url().includes('_data/isomorphism.json'), { timeout: 20000 }
        ).catch(() => null);
        await page.selectOption('#graph-select', 'isomorphism');
        const isoResp = await isoPromise;
        await waitLoaded(page);
        const s4 = await readState(page);
        const ink4 = await canvasInk(page);
        const ok4 = !!isoResp && isoResp.status() === 200 && s4.graphKey === 'isomorphism' &&
            s4.datasetLength > 0 && s4.visibleNodes > 0 && ink4.ok && ink4.fraction > 0.005 &&
            cleanSince(snap4);
        record(4, 'Isomorphism graph renders with no errors', ok4,
            `isomorphism.json -> HTTP ${isoResp ? isoResp.status() : 'NOT REQUESTED'}; ` +
            `${s4.datasetLength} nodes, ${s4.edgeLength} edges, title="${s4.title}", ` +
            `${ink4.ok ? (ink4.fraction * 100).toFixed(1) + '% non-bg pixels' : ink4.reason}\n` +
            (describeSince(snap4).join('\n') || 'no errors'));

        // ---------------------------------------------------------------- 5
        phase = 'evidence-search-click';
        const snap5 = snapshotErrors();
        await page.selectOption('#graph-select', 'evidence');
        await waitLoaded(page);

        // Pick a real id straight out of entities.json rather than hard-coding.
        const entities = JSON.parse(await fsp.readFile(path.join(ROOT, '_data', 'entities.json'), 'utf8'));
        const target = (entities.claims || []).find(c => /^f\d+-/.test(c.id) && c.status) ||
            (entities.claims || [])[0];
        if (!target) throw new Error('entities.json has no claims to search for');

        await typeSearch(page, target.id);
        const s5 = await readState(page);
        const searchOk = s5.visibleIds.includes(target.id) &&
            s5.visibleNodes >= 1 && s5.visibleNodes < evidenceCount;

        const clickPt = await clickNode(page, target.id);
        const detail = await page.evaluate(() => {
            const panel = document.getElementById('details-panel');
            const pill = panel ? panel.querySelector('.status-pill') : null;
            return {
                open: !!panel && !panel.classList.contains('hidden'),
                title: (document.getElementById('detail-title') || {}).textContent || '',
                pill: pill ? pill.textContent.trim() : null,
                hasChainBtn: !!document.getElementById('btn-copy-chain'),
                selected: typeof currentNodeId !== 'undefined' ? currentNodeId : null
            };
        });
        const clickOk = detail.open && detail.selected === target.id && !!detail.pill;
        record(5, 'search surfaces a known id; clicking the node opens details with a status pill',
            searchOk && clickOk && cleanSince(snap5),
            `search "${target.id}" -> ${s5.visibleNodes}/${evidenceCount} nodes visible, ` +
            `target present=${s5.visibleIds.includes(target.id)}\n` +
            `real mouse click at (${clickPt.x.toFixed(0)},${clickPt.y.toFixed(0)}) -> ` +
            `panel open=${detail.open}, currentNodeId=${detail.selected}, ` +
            `status pill=${detail.pill === null ? 'MISSING' : '"' + detail.pill + '"'}, ` +
            `title="${detail.title}"\n` +
            (describeSince(snap5).join('\n') || 'no errors'));

        // ---------------------------------------------------------------- 6
        phase = 'timeline';
        const snap6 = snapshotErrors();
        await typeSearch(page, '');           // clear the search filter first
        await page.waitForTimeout(200);
        const beforeTimeline = await readState(page);

        const slider = await page.evaluate(() => {
            const el = document.getElementById('timeline');
            const bar = document.getElementById('timeline-bar');
            return el ? {
                min: el.min, max: el.max, value: el.value,
                barHidden: !!bar && bar.classList.contains('mode-hidden'),
                disabled: !!bar && bar.classList.contains('disabled')
            } : null;
        });
        if (!slider) throw new Error('#timeline slider is not in the DOM');

        async function setSlider(v) {
            await page.evaluate((val) => {
                const el = document.getElementById('timeline');
                el.value = String(val);
                el.dispatchEvent(new Event('input', { bubbles: true }));
            }, v);
            await page.waitForTimeout(250);
            return readState(page);
        }

        const atMin = await setSlider(slider.min);
        const atMax = await setSlider(slider.max);
        const timelineOk = Number(slider.max) > Number(slider.min) &&
            atMin.visibleNodes < beforeTimeline.visibleNodes &&
            atMax.visibleNodes === beforeTimeline.visibleNodes &&
            atMax.datasetLength === beforeTimeline.datasetLength &&
            cleanSince(snap6);
        record(6, 'timeline scrubber hides nodes at minimum and restores the full count at maximum',
            timelineOk,
            `slider range [${slider.min}..${slider.max}] (DataSet.length=${beforeTimeline.datasetLength})\n` +
            `full=${beforeTimeline.visibleNodes} visible -> min=${atMin.visibleNodes} -> ` +
            `max=${atMax.visibleNodes}\n` +
            (describeSince(snap6).join('\n') || 'no errors'));

        // ---------------------------------------------------------------- 7
        phase = 'copy-chain';
        const snap7 = snapshotErrors();
        await typeSearch(page, target.id);
        await clickNode(page, target.id);
        const chain = await page.evaluate((id) => {
            const btn = document.getElementById('btn-copy-chain');
            const out = {
                btnPresent: !!btn,
                btnLabel: btn ? btn.textContent.trim() : null,
                text: null,
                clicked: false,
                toast: null
            };
            if (typeof buildEvidenceChainText !== 'function') return out;
            out.text = buildEvidenceChainText(id, 3);
            if (btn) { btn.click(); out.clicked = true; }
            return out;
        }, target.id);
        await page.waitForTimeout(400);
        const toast = await page.evaluate(() => {
            const t = document.getElementById('toast');
            return t ? { text: t.textContent, visible: t.classList.contains('visible') } : null;
        });
        const chainOk = chain.btnPresent && typeof chain.text === 'string' &&
            chain.text.trim().length > 0 && chain.clicked && cleanSince(snap7);
        record(7, '"copy evidence chain" control produces a non-empty chain string', chainOk,
            `control "${chain.btnLabel}" present=${chain.btnPresent}, clicked=${chain.clicked}\n` +
            `toast: ${toast ? JSON.stringify(toast) : 'none'}\n` +
            `chain (${chain.text ? chain.text.length : 0} chars, ` +
            `${chain.text ? chain.text.split('\n').length : 0} lines):\n` +
            (chain.text || '(NONE)').split('\n').slice(0, 8).map(l => '  | ' + l).join('\n') +
            (chain.text && chain.text.split('\n').length > 8 ? '\n  | ...' : '') + '\n' +
            (describeSince(snap7).join('\n') || 'no errors'));

        // ---------------------------------------------------------------- 8
        phase = 'summary';
        const ok8 = badResponses.length === 0;
        const byStatus = {};
        allResponses.forEach(r => { byStatus[r.status] = (byStatus[r.status] || 0) + 1; });
        record(8, 'no HTTP 404s (or any >=400) for any requested asset', ok8,
            `${allResponses.length} responses recorded: ` +
            Object.entries(byStatus).sort().map(([s, c]) => `${s}x${c}`).join(', ') +
            (ok8 ? '' : '\n' + badResponses.map(r => `  ${r.status} [${r.phase}] ${r.url}`).join('\n')));

    } catch (err) {
        console.error('\n\x1b[31mHARNESS ERROR\x1b[0m:', err && err.stack || err);
        exitCode = 1;
    } finally {
        await context.close().catch(() => {});
        await browser.close().catch(() => {});
        await new Promise(r => server.close(r));
    }

    // --- summary -----------------------------------------------------------
    const failed = results.filter(r => !r.ok);
    console.log('\n--- summary ---');
    for (const r of results) console.log(`  ${r.ok ? 'PASS' : 'FAIL'}  ${r.n}. ${r.title}`);
    console.log(`\n${results.length - failed.length}/${results.length} assertions passed` +
        (results.length < 8 ? `  (only ${results.length}/8 assertions ran)` : ''));
    if (pageErrors.length) {
        console.log(`\nAll pageerrors (${pageErrors.length}):`);
        pageErrors.forEach(e => console.log(`  [${e.phase}] ${e.text}`));
    }
    if (consoleErrors.length) {
        console.log(`\nAll console errors (${consoleErrors.length}):`);
        consoleErrors.forEach(e => console.log(`  [${e.phase}] ${e.text}`));
    }
    if (badResponses.length) {
        console.log(`\nAll >=400 responses (${badResponses.length}):`);
        badResponses.forEach(e => console.log(`  ${e.status} [${e.phase}] ${e.url}`));
    }
    if (failed.length || results.length < 8) exitCode = 1;
    process.exit(exitCode);
}

main();
