#!/usr/bin/env node
/**
 * Headless browser smoke + interaction test for docs/graph/viewer.html.
 *
 * Serves docs/graph on a free port, drives the viewer with Playwright's
 * bundled Chromium, and asserts fourteen things:
 *
 *   1. viewer.html loads with ZERO pageerror and ZERO console-error events
 *   2. a <canvas> exists and vis-network has actually rendered nodes
 *   3. Dissolution graph loads dissolution.json + re-renders on model switch
 *   4. Isomorphism graph renders
 *   5. search surfaces a known id; clicking that node opens details + status pill
 *   6. the timeline scrubber hides nodes at min and restores them at max
 *   7. "copy evidence chain" produces a non-empty chain string
 *   8. the "not-supported" status chip filters to exactly its nodes, in #8F5A57
 *   9. the legend carries "not-supported" in its severity-gradient position
 *  10. an unknown status still renders, gets a chip, and stays reachable
 *  11. no HTTP >= 400 responses for any requested asset
 *  12. on a phone-sized viewport the graph, not the chrome, owns the screen
 *  13. returning to evidence from another graph restores force-directed layout
 *  14. Threads mode renders from threads.json — readiness colours reach the
 *      DataSet, the ranked low-hanging-fruit list is populated and drives the
 *      graph, blocks/blocked-by are drawn as gate edges — and none of it
 *      leaks: after threads -> dissolution -> evidence the hub nodes are gone,
 *      status colours are back, gate edges are undashed and evidence is still
 *      clickable
 *
 * Assertions 1-11, 13 and 14 run at 1600x1000. Assertion 12 opens a second,
 * phone-sized context (393x830, isMobile + hasTouch) so the narrow-screen
 * layout is pinned without disturbing the desktop measurements the others
 * depend on.
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
            // Never serve outside the root. The separator matters: a bare
            // startsWith(rootDir) also admits a *sibling* directory whose name
            // begins with the root's (<root>-other/secrets).
            if (abs !== rootDir && !abs.startsWith(rootDir + path.sep)) {
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
// Every assertion must run: a suite that silently stops short is a failure, not
// a pass. Bump this when adding one.
const TOTAL_ASSERTIONS = 14;

// --- mobile budget (assertion 12) ------------------------------------------
// A real device (Nothing Phone 2a) showed the header + chip rows eating 66% of
// the screen. These are the numbers that made the graph usable again; they are
// a budget, not a description, so tightening the layout is free but loosening
// it has to be a deliberate edit here.
const MOBILE_VIEWPORT = { width: 393, height: 830 };
const MAX_CHROME_PX = 160;     // everything above #graph
const MIN_GRAPH_SHARE = 0.60;  // #graph height / viewport height
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
        // "not-supported" is the eighth status, added after this suite was
        // written, so nothing above covers it. Assertions 8-10 pin the whole
        // path — chip, filter, fill colour, legend — plus the unknown-status
        // fallback that keeps older/newer vocabularies from losing nodes.
        phase = 'not-supported-status';
        const snap8 = snapshotErrors();
        await typeSearch(page, '');                 // drop the test-7 search filter
        await page.evaluate(() => {
            const el = document.getElementById('timeline');
            if (el) { el.value = el.max; el.dispatchEvent(new Event('input', { bubbles: true })); }
            setAllStatusChips(true);
        });
        await page.waitForTimeout(250);

        const NOT_SUPPORTED = 'not-supported';
        const NOT_SUPPORTED_HEX = '#8F5A57';
        const DEFAULT_HEX = '#7F8C8D';
        // Ground truth is the regenerated entities.json, never the page.
        const expectedNotSupported = (entities.claims || [])
            .filter(c => c.status === NOT_SUPPORTED).map(c => c.id).sort();

        const chipState = await page.evaluate((st) => {
            const chip = document.getElementById('chip-status-' + st);
            const order = typeof STATUS_ORDER !== 'undefined' ? STATUS_ORDER : [];
            return {
                exists: !!chip,
                text: chip ? chip.textContent.trim() : null,
                order,
                afterRefuted: order.indexOf(st) >= 0 &&
                    order.indexOf(st) === order.indexOf('refuted') + 1,
                palette: typeof statusColour === 'function' ? statusColour(st) : null
            };
        }, NOT_SUPPORTED);

        // Click the chip for real with every other status off. Runs, models and
        // sources carry no status and are deliberately never gated by the status
        // chips, so the exact-match check is over status-carrying nodes only.
        const filtered = await page.evaluate((st) => {
            setAllStatusChips(false);
            const chip = document.getElementById('chip-status-' + st);
            if (chip) chip.click();                 // exercises the real onclick
            const byId = {};
            allNodes.forEach(n => { byId[n.id] = n; });
            const items = nodesDataSet.get();
            const colours = {};
            items.filter(n => byId[n.id] && byId[n.id].status === st)
                .forEach(n => { colours[n.id] = n.color.background; });
            return {
                clickable: !!chip,
                chipActive: !!chip && chip.classList.contains('active'),
                visibleWithStatus: items
                    .filter(n => n.hidden !== true && byId[n.id] && byId[n.id].status)
                    .map(n => n.id).sort(),
                otherStatusesHidden: items
                    .filter(n => n.hidden !== true && byId[n.id] &&
                        byId[n.id].status && byId[n.id].status !== st).length,
                colours
            };
        }, NOT_SUPPORTED);

        const colourVals = Object.values(filtered.colours).map(c => String(c).toUpperCase());
        // A dataset carrying no not-supported claims is a legitimate case (that is
        // the backward-compatibility requirement), so the per-node colour check is
        // conditional on there being nodes. The vocabulary checks — chip, gradient
        // position, palette entry — are unconditional and hold for any dataset.
        const colourOk = colourVals.length === expectedNotSupported.length &&
            colourVals.every(c => c === NOT_SUPPORTED_HEX && c !== DEFAULT_HEX);
        const sameSet = JSON.stringify(filtered.visibleWithStatus) ===
            JSON.stringify(expectedNotSupported);
        const ok8 = chipState.exists && chipState.afterRefuted &&
            String(chipState.palette).toUpperCase() === NOT_SUPPORTED_HEX &&
            filtered.clickable && filtered.chipActive &&
            sameSet && filtered.otherStatusesHidden === 0 && colourOk &&
            cleanSince(snap8);
        record(8, '"not-supported" chip filters to exactly its nodes, drawn in #8F5A57', ok8,
            `chip present=${chipState.exists} text="${chipState.text}" ` +
            `clickable=${filtered.clickable} active-after-click=${filtered.chipActive}\n` +
            `STATUS_ORDER=[${chipState.order.join(', ')}] -> immediately after "refuted"=` +
            `${chipState.afterRefuted}\n` +
            `entities.json says not-supported = [${expectedNotSupported.join(', ')}] ` +
            `(${expectedNotSupported.length}); status-carrying nodes visible after filtering = ` +
            `[${filtered.visibleWithStatus.join(', ')}] (${filtered.visibleWithStatus.length}) ` +
            `-> match=${sameSet}, other-status leakage=${filtered.otherStatusesHidden}\n` +
            `rendered fills: ${JSON.stringify(filtered.colours)} ` +
            `(expected ${NOT_SUPPORTED_HEX}, must not be default ${DEFAULT_HEX})\n` +
            (describeSince(snap8).join('\n') || 'no errors'));

        // ---------------------------------------------------------------- 9
        phase = 'not-supported-legend';
        const snap9 = snapshotErrors();
        const legend = await page.evaluate((st) => {
            setAllStatusChips(true);
            if (typeof setColourMode === 'function') setColourMode('status');
            const toHex = (rgb) => {
                const m = String(rgb).match(/(\d+)\D+(\d+)\D+(\d+)/);
                if (!m) return String(rgb).toUpperCase();
                return '#' + [1, 2, 3].map(i => Number(m[i]).toString(16).padStart(2, '0'))
                    .join('').toUpperCase();
            };
            const rows = [...document.querySelectorAll('#legend .legend-row')]
                .map(r => ({
                    label: r.textContent.trim(),
                    swatch: r.querySelector('.legend-swatch')
                        ? toHex(r.querySelector('.legend-swatch').style.background) : null
                }));
            const statusRows = rows.slice(0, rows.findIndex(r => r.label === 'untested') + 1);
            const idx = statusRows.findIndex(r => r.label === st);
            return {
                statusRows,
                idx,
                refutedIdx: statusRows.findIndex(r => r.label === 'refuted'),
                swatch: idx >= 0 ? statusRows[idx].swatch : null
            };
        }, NOT_SUPPORTED);
        // The legend lists only statuses the loaded data actually carries, so on a
        // dataset with no not-supported claims the correct expectation is that the
        // row is absent and the other seven are unchanged — that is the
        // "renders identically" half of backward compatibility.
        const legendExpected = expectedNotSupported.length > 0;
        const ok9 = legendExpected
            ? (legend.idx >= 0 && legend.idx === legend.refutedIdx + 1 &&
               legend.swatch === NOT_SUPPORTED_HEX && cleanSince(snap9))
            : (legend.idx === -1 && legend.refutedIdx >= 0 && cleanSince(snap9));
        record(9, 'legend carries a "not-supported" entry, in gradient position, correct swatch', ok9,
            `legend status block: ` +
            legend.statusRows.map(r => `${r.label}=${r.swatch}`).join(', ') + '\n' +
            (legendExpected
                ? `"${NOT_SUPPORTED}" at index ${legend.idx} (refuted at ${legend.refutedIdx}), ` +
                  `swatch=${legend.swatch} (expected ${NOT_SUPPORTED_HEX})`
                : `dataset carries no ${NOT_SUPPORTED} claims; legend correctly omits the row ` +
                  `(idx=${legend.idx}) and keeps the other statuses intact`) + '\n' +
            (describeSince(snap9).join('\n') || 'no errors'));

        // ---------------------------------------------------------------- 10
        // Forward compatibility: a status this build of the viewer has never
        // heard of must still get a chip, a legend row, the default fill, and
        // must stay visible and searchable. Injected into the in-memory model
        // only — the committed JSON is not touched.
        phase = 'unknown-status-fallback';
        const snap10 = snapshotErrors();
        const UNKNOWN_STATUS = 'quantum-undecided';
        const injected = await page.evaluate((st) => {
            const victim = allNodes.find(n => n.status === 'supported');
            if (!victim) return { error: 'no supported node to re-label' };
            const original = victim.status;
            victim.status = st;
            buildChips(); syncChipClasses(); applyVisualState();
            const chip = document.getElementById('chip-status-' + st);
            const item = nodesDataSet.get(victim.id);
            const legendLabels = [...document.querySelectorAll('#legend .legend-row')]
                .map(r => r.textContent.trim());
            return {
                id: victim.id,
                original,
                chipExists: !!chip,
                chipText: chip ? chip.textContent.trim() : null,
                chipAppendedLast: !!chip && chip === [...document.querySelectorAll(
                    '#status-chips .chip[data-status]')].pop(),
                hidden: item.hidden === true,
                fill: String(item.color.background).toUpperCase(),
                legendHasIt: legendLabels.includes(st),
                // Object.prototype keys must not be mistaken for palette entries.
                protoSafe: statusColour('constructor') === statusColour('toString') &&
                    String(statusColour('constructor')).toUpperCase() === '#7F8C8D'
            };
        }, UNKNOWN_STATUS);

        let reach = { visible: 0, includes: false };
        if (injected.id) {
            await typeSearch(page, injected.id);
            reach = await page.evaluate((id) => {
                const items = nodesDataSet.get().filter(n => n.hidden !== true).map(n => n.id);
                return { visible: items.length, includes: items.includes(id) };
            }, injected.id);
            await typeSearch(page, '');
            // Put the in-memory model back the way we found it.
            await page.evaluate((o) => {
                const n = allNodes.find(x => x.id === o.id);
                if (n) n.status = o.original;
                buildChips(); syncChipClasses(); applyVisualState();
            }, { id: injected.id, original: injected.original });
        }
        const ok10 = !injected.error && injected.chipExists && injected.chipAppendedLast &&
            injected.hidden === false && injected.fill === DEFAULT_HEX &&
            injected.legendHasIt && injected.protoSafe &&
            reach.includes && cleanSince(snap10);
        record(10, 'a status the viewer has never seen still renders, filters and is reachable', ok10,
            `injected status="${UNKNOWN_STATUS}" onto ${injected.id} (was "${injected.original}") ` +
            `via page.evaluate; entities.json untouched\n` +
            `chip="${injected.chipText}" appended after the canonical eight=` +
            `${injected.chipAppendedLast}; legend row present=${injected.legendHasIt}\n` +
            `node hidden=${injected.hidden}, fill=${injected.fill} ` +
            `(expected default ${DEFAULT_HEX})\n` +
            `prototype-key safety (statusColour('constructor')/('toString') -> default)=` +
            `${injected.protoSafe}\n` +
            `search "${injected.id}" -> ${reach.visible} visible, target present=${reach.includes}\n` +
            (describeSince(snap10).join('\n') || 'no errors'));

        // ---------------------------------------------------------------- 11
        phase = 'summary';
        const ok11 = badResponses.length === 0;
        const byStatus = {};
        allResponses.forEach(r => { byStatus[r.status] = (byStatus[r.status] || 0) + 1; });
        record(11, 'no HTTP 404s (or any >=400) for any requested asset', ok11,
            `${allResponses.length} responses recorded: ` +
            Object.entries(byStatus).sort().map(([s, c]) => `${s}x${c}`).join(', ') +
            (ok11 ? '' : '\n' + badResponses.map(r => `  ${r.status} [${r.phase}] ${r.url}`).join('\n')));

        // ---------------------------------------------------------------- 12
        // Phone layout. A separate context so assertions 1-11 keep measuring
        // the desktop layout they were written against.
        phase = 'mobile-layout';
        const snap12 = snapshotErrors();
        const mobileContext = await browser.newContext({
            viewport: MOBILE_VIEWPORT,
            deviceScaleFactor: 2.75,
            isMobile: true,
            hasTouch: true
        });
        const mobileRows = [];
        try {
            const mPage = await mobileContext.newPage();
            await mPage.route('**/*', async (route) => {
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
            mPage.on('pageerror', (err) => {
                pageErrors.push({ phase, text: (err && err.stack) || String(err) });
            });
            mPage.on('console', (msg) => {
                if (msg.type() === 'error') consoleErrors.push({ phase, text: msg.text() });
            });
            mPage.on('response', (res) => {
                const rec = { phase, status: res.status(), url: res.url() };
                allResponses.push(rec);
                if (rec.status >= 400) badResponses.push(rec);
            });

            await mPage.goto(`${base}/viewer.html`, { waitUntil: 'domcontentloaded' });
            await waitLoaded(mPage);

            // Measure the resting layout, then again with the Filters
            // disclosure open: opening it must overlay the graph, never push it
            // down. Both states are held to the same budget.
            const measure = () => mPage.evaluate(() => {
                const g = document.getElementById('graph').getBoundingClientRect();
                return {
                    chrome: Math.round(g.top),
                    graphHeight: Math.round(g.height),
                    graphWidth: Math.round(g.width),
                    viewportHeight: window.innerHeight,
                    viewportWidth: window.innerWidth,
                    scrollWidth: document.documentElement.scrollWidth
                };
            });

            // The disclosure is the mechanism the budget depends on, so a
            // missing toggle is a failure in its own right -- but report it as
            // one instead of letting page.click() time out and abort the run.
            const toggle = () => mPage.evaluate(() => {
                const el = document.getElementById('mobile-filter-toggle');
                if (!el || getComputedStyle(el).display === 'none') return false;
                el.click();
                return true;
            });

            for (const key of ['evidence', 'dissolution', 'isomorphism']) {
                phase = `mobile-layout-${key}`;
                if (key !== 'evidence') {
                    await mPage.selectOption('#graph-select', key);
                    await waitLoaded(mPage);
                }
                for (const filtersOpen of [false, true]) {
                    let toggled = true;
                    if (filtersOpen) {
                        toggled = await toggle();
                        await sleep(400);
                    }
                    const m = await measure();
                    mobileRows.push({
                        key,
                        filtersOpen,
                        toggled,
                        ...m,
                        share: m.graphHeight / m.viewportHeight,
                        chromeOk: m.chrome <= MAX_CHROME_PX,
                        shareOk: m.graphHeight / m.viewportHeight >= MIN_GRAPH_SHARE,
                        widthOk: m.scrollWidth <= m.viewportWidth
                    });
                    if (filtersOpen && toggled) {
                        await toggle();
                        await sleep(300);
                    }
                }
            }
        } finally {
            await mobileContext.close().catch(() => {});
        }
        phase = 'mobile-layout';
        const ok12 = mobileRows.length === 6 &&
            mobileRows.every(r => r.chromeOk && r.shareOk && r.widthOk && r.toggled) &&
            cleanSince(snap12);
        record(12,
            `on a ${MOBILE_VIEWPORT.width}x${MOBILE_VIEWPORT.height} phone the graph owns the ` +
            `screen (chrome <= ${MAX_CHROME_PX}px, graph >= ${Math.round(MIN_GRAPH_SHARE * 100)}% ` +
            `of viewport height, no horizontal scroll)`,
            ok12,
            (mobileRows.length
                ? mobileRows.map(r =>
                    `${r.key.padEnd(12)} filters=${r.filtersOpen ? 'open  ' : 'closed'} ` +
                    `chrome=${String(r.chrome).padStart(3)}px${r.chromeOk ? '' : ' <-- OVER BUDGET'} ` +
                    `graph=${r.graphWidth}x${r.graphHeight} ` +
                    `(${(r.share * 100).toFixed(1)}%${r.shareOk ? '' : ' <-- UNDER BUDGET'}) ` +
                    `scrollWidth=${r.scrollWidth}/${r.viewportWidth}` +
                    `${r.widthOk ? '' : ' <-- HORIZONTAL SCROLL'}` +
                    `${r.toggled ? '' : ' <-- no usable #mobile-filter-toggle'}`).join('\n')
                : 'no measurements taken') +
            '\n' + (describeSince(snap12).join('\n') || 'no errors'));

        // ------------------------------------------------------------------
        // 13. Returning from dissolution restores the force-directed layout.
        //
        // Dissolution renders hierarchically with physics off. That override
        // used to survive the trip back, so the evidence graph re-rendered as
        // hierarchical columns with physics disabled — 163 nodes collapsed onto
        // 9 distinct x positions instead of ~155. The distinct-x count is the
        // signal that actually catches it: the mode flags could be restored
        // while stale pinned positions still left the graph stacked.
        // ------------------------------------------------------------------
        phase = 'layout-restore';
        const snap13 = snapshotErrors();

        const readLayout = () => page.evaluate(() => {
            const xs = Object.values(network.getPositions()).map(p => Math.round(p.x));
            return {
                hierarchical: !!network.layoutEngine.options.hierarchical.enabled,
                physics: !!network.physics.options.enabled,
                overrideCleared: typeof layoutOverride === 'undefined' || layoutOverride === null,
                distinctX: new Set(xs).size,
                nodes: xs.length
            };
        });

        await page.selectOption('#graph-select', 'evidence');
        await waitLoaded(page);
        await sleep(1500);
        const baseline = await readLayout();

        const roundTrips = [];
        for (const via of ['dissolution', 'isomorphism']) {
            await page.selectOption('#graph-select', via);
            await waitLoaded(page);
            await sleep(1200);
            const detour = await readLayout();

            await page.selectOption('#graph-select', 'evidence');
            await waitLoaded(page);
            await sleep(1500);
            const back = await readLayout();

            roundTrips.push({
                via,
                detour,
                back,
                ok: !back.hierarchical && back.physics && back.overrideCleared &&
                    back.nodes === baseline.nodes &&
                    // Allow generous drift: physics is stochastic run to run, but
                    // the failure mode collapses this by an order of magnitude.
                    back.distinctX >= Math.floor(baseline.distinctX * 0.5)
            });
        }

        const ok13 = !baseline.hierarchical && baseline.physics &&
            roundTrips.length === 2 && roundTrips.every(r => r.ok) && cleanSince(snap13);
        record(13,
            'returning to evidence from another graph restores force-directed layout ' +
            '(hierarchical off, physics on, nodes not stacked)',
            ok13,
            `baseline evidence: hierarchical=${baseline.hierarchical} physics=${baseline.physics} ` +
            `distinctX=${baseline.distinctX}/${baseline.nodes}\n` +
            roundTrips.map(r =>
                `via ${r.via.padEnd(12)} -> back: hierarchical=${r.back.hierarchical}` +
                `${r.back.hierarchical ? ' <-- LEAKED' : ''} ` +
                `physics=${r.back.physics}${r.back.physics ? '' : ' <-- DISABLED'} ` +
                `overrideCleared=${r.back.overrideCleared} ` +
                `distinctX=${r.back.distinctX}/${r.back.nodes}` +
                `${r.ok ? '' : ' <-- STACKED'}`).join('\n') +
            '\n' + (describeSince(snap13).join('\n') || 'no errors'));

        // ------------------------------------------------------------------
        // 14. Threads mode renders, and leaves nothing behind.
        //
        // Threads is an overlay view: the same evidence nodes re-coloured by
        // _data/threads.json, plus synthesised blocker hubs and loudly-styled
        // blocks / blocked-by edges. Three things have to hold at once.
        //
        //   (a) it renders at all: readiness colours actually reach the vis
        //       DataSet (not just the legend), the ranked low-hanging-fruit
        //       list is populated from the report, and clicking a ranked row
        //       focuses the graph the way a node click does;
        //   (b) the gate edges are drawn as gate edges — dashed, width 4, in
        //       the gate colour — since a blocker gating several claims
        //       reading as a hub is the whole reason the view exists;
        //   (c) and, critically, none of that survives the trip out. Threads
        //       adds nodes (hubs), rewrites node colours, and mutates edge
        //       objects in place. Every one of those is a leak waiting to
        //       happen: come back to evidence via dissolution and the graph
        //       could keep the hub nodes, keep readiness colours instead of
        //       status colours, or keep drawing blocks edges dashed. Assertion
        //       13 covers layout leaking; this covers *content* leaking, and
        //       goes out through dissolution so a two-hop return is tested.
        // ------------------------------------------------------------------
        phase = 'threads';
        const snap14 = snapshotErrors();

        const THREADS_TITLE =
            'Threads mode renders from threads.json (readiness colours, ranked list, ' +
            'gate edges) and leaves no trace after threads -> dissolution -> evidence';

        // _data/threads.json is a build product. The viewer treats a missing
        // overlay as degraded-but-valid, so the suite must too: an unreadable
        // file is a recorded failure of assertion 14 with an actionable message,
        // never an exception into the outer catch — that would abort the run and
        // skip every assertion after this one.
        const threadsPath = path.join(ROOT, '_data', 'threads.json');
        let report = null;
        let threadsLoadError = null;
        try {
            report = JSON.parse(await fsp.readFile(threadsPath, 'utf8'));
        } catch (err) {
            threadsLoadError = err;
        }

        if (!report) {
            record(14, THREADS_TITLE, false,
                `cannot read the threads overlay at ${threadsPath}\n` +
                `  ${(threadsLoadError && threadsLoadError.message) || 'file is empty'}\n` +
                'generate it with `python3 docs/graph/build_threads_report.py` (or point ' +
                'GRAPH_ROOT at a docs/graph copy that has _data/threads.json) and re-run.');
        } else {
            const reportFruit = (report.low_hanging_fruit || []).length;
            const graphGateEdges = (entities.relationships || [])
                .filter(r => r.type === 'blocks' || r.type === 'blocked-by').length;

            // Everything threads-specific, read out of the live page.
            //
            // Note on where things are read from: `edgesDataSet` is a stripped copy
            // (createEdges' `relType`/`data` are deleted before it is handed to
            // vis), so edge *identity* comes from `allEdges` and edge *styling*
            // from the DataSet row with the same id. Same split for nodes: `status`
            // lives on `allNodes`, the painted colour on `nodesDataSet`.
            const readThreads = () => page.evaluate(() => {
                const dsNodes = (typeof nodesDataSet !== 'undefined' && nodesDataSet) ? nodesDataSet.get() : [];
                const dsEdges = (typeof edgesDataSet !== 'undefined' && edgesDataSet) ? edgesDataSet.get() : [];
                const drawnEdge = {};
                dsEdges.forEach(e => { drawnEdge[e.id] = e; });
                const gateTypes = ['blocks', 'blocked-by'];
                const gate = (typeof allEdges !== 'undefined' ? allEdges : [])
                    .filter(e => gateTypes.indexOf(e.relType) !== -1);
                const styled = gate.filter(e => {
                    const d = drawnEdge[e.id];
                    return d && Array.isArray(d.dashes) && d.dashes.length > 0 &&
                        Number(d.width) === 4 &&
                        d.color && String(d.color.color).toUpperCase() === '#6B4C8A';
                });
                const paint = {};
                dsNodes.forEach(n => {
                    paint[n.id] = (n.color && n.color.background)
                        ? String(n.color.background).toUpperCase() : null;
                });
                const byReadiness = {};
                const colourByReadiness = {};
                (typeof allNodes !== 'undefined' ? allNodes : []).forEach(n => {
                    const key = (typeof readinessOf === 'function') ? readinessOf(n) : 'neutral';
                    byReadiness[key] = (byReadiness[key] || 0) + 1;
                    if (!colourByReadiness[key]) colourByReadiness[key] = paint[n.id];
                });
                const nodes = dsNodes;
                const edges = dsEdges;
                const panel = document.getElementById('fruit-panel');
                const rows = panel ? Array.from(panel.querySelectorAll('.fruit-row')) : [];
                return {
                    renderMode: typeof renderMode !== 'undefined' ? renderMode : null,
                    nodeCount: nodes.length,
                    edgeCount: edges.length,
                    hubCount: (typeof threadsHubs !== 'undefined' && threadsHubs) ? threadsHubs.length : 0,
                    reportLoaded: typeof threadsReport !== 'undefined' && !!threadsReport,
                    fruitInState: (typeof threadsFruit !== 'undefined' && threadsFruit) ? threadsFruit.length : 0,
                    panelVisible: !!panel && !panel.classList.contains('mode-hidden'),
                    fruitRows: rows.length,
                    firstRowText: rows.length ? rows[0].textContent.replace(/\s+/g, ' ').trim() : null,
                    gateEdges: gate.length,
                    gateStyled: styled.length,
                    readinessChipsVisible: (() => {
                        const el = document.getElementById('readiness-chips');
                        return !!el && !el.classList.contains('mode-hidden');
                    })(),
                    byReadiness,
                    colourByReadiness,
                    answeredColour: colourByReadiness['answered-unrecorded'] || null,
                    canvases: document.querySelectorAll('#graph canvas').length
                };
            });

            // Evidence-mode facts that threads must not have disturbed.
            const readEvidenceContent = (probeId) => page.evaluate((probe) => {
                const dsNodes = (typeof nodesDataSet !== 'undefined' && nodesDataSet) ? nodesDataSet.get() : [];
                const dsEdges = (typeof edgesDataSet !== 'undefined' && edgesDataSet) ? edgesDataSet.get() : [];
                const drawnEdge = {};
                dsEdges.forEach(e => { drawnEdge[e.id] = e; });
                // Evidence mode has its own blocks / blocked-by style, supplied by
                // _data/visual_config.json -- these edges are legitimately dashed
                // here. The leak to catch is the *threads* gate style (its own
                // colour and dash pattern) surviving the trip back, so compare the
                // drawn style against the config rather than against "undashed".
                const gate = (typeof allEdges !== 'undefined' ? allEdges : [])
                    .filter(e => e.relType === 'blocks' || e.relType === 'blocked-by');
                const gateStyle = gate.map(e => {
                    const d = drawnEdge[e.id];
                    return {
                        id: e.id, type: e.relType, drawn: !!d,
                        colour: d && d.color ? String(d.color.color).toUpperCase() : null,
                        dashes: d ? JSON.stringify(d.dashes) : null,
                        width: d ? Number(d.width) : null
                    };
                });
                const wearingThreadsStyle = gateStyle.filter(g =>
                    g.colour === String(GATE_EDGE_STYLE.color).toUpperCase() ||
                    g.dashes === JSON.stringify(GATE_EDGE_STYLE.dashes));
                const paint = {};
                dsNodes.forEach(n => {
                    paint[n.id] = (n.color && n.color.background)
                        ? String(n.color.background).toUpperCase() : null;
                });
                // Every node carrying a status must be painted the colour that
                // status implies -- or the retired grey, if the timeline cursor has
                // already passed its retirement. Nothing else is acceptable.
                const wrong = [];
                let checked = 0;
                (typeof allNodes !== 'undefined' ? allNodes : []).forEach(n => {
                    if (!n.status || typeof statusColour !== 'function') return;
                    checked++;
                    const want = String(statusColour(n.status)).toUpperCase();
                    const retired = (typeof RETIRED_COLOR !== 'undefined')
                        ? String(RETIRED_COLOR).toUpperCase() : null;
                    const got = paint[n.id];
                    if (got !== want && got !== retired) {
                        wrong.push({ id: n.id, status: n.status, want, got });
                    }
                });
                // The motivating node: red for "answered, unrecorded" in threads,
                // and it must be back to its epistemic status colour here.
                const probeNode = (typeof allNodes !== 'undefined' ? allNodes : [])
                    .find(n => n.id === probe) || null;
                return {
                    renderMode: typeof renderMode !== 'undefined' ? renderMode : null,
                    nodeCount: dsNodes.length,
                    edgeCount: dsEdges.length,
                    hubNodes: dsNodes.filter(n => String(n.id).indexOf('threads-blocker-') === 0).length,
                    gateEdges: gate.length,
                    gateWearingThreadsStyle: wearingThreadsStyle.length,
                    gateStyleSample: gateStyle.slice(0, 2),
                    statusChecked: checked,
                    statusWrong: wrong.slice(0, 5),
                    statusWrongCount: wrong.length,
                    probeId: probe,
                    probeStatus: probeNode ? probeNode.status : null,
                    probePaint: paint[probe] || null,
                    probeWant: (probeNode && typeof statusColour === 'function')
                        ? String(statusColour(probeNode.status)).toUpperCase() : null,
                    readinessChipsVisible: (() => {
                        const el = document.getElementById('readiness-chips');
                        return !!el && !el.classList.contains('mode-hidden');
                    })(),
                    fruitPanelVisible: (() => {
                        const el = document.getElementById('fruit-panel');
                        return !!el && !el.classList.contains('mode-hidden');
                    })(),
                    threadsStateCleared: (typeof threadsReport === 'undefined' || threadsReport === null) &&
                        (typeof threadsFruit === 'undefined' || threadsFruit.length === 0)
                };
            }, probeId);

            await page.selectOption('#graph-select', 'threads');
            await waitLoaded(page);
            await sleep(800);
            const th = await readThreads();

            // (a) the ranked list is real and clicking it drives the graph.
            const beforeFocus = await readState(page);
            const firstFruit = (report.low_hanging_fruit || [])[0] || {};
            let focusVisible = null, focusSelected = null, focusPill = null;
            if (th.fruitRows > 0) {
                await page.click('#fruit-panel .fruit-row:first-of-type');
                await page.waitForTimeout(600);
                const after = await readState(page);
                focusVisible = after.visibleNodes;
                const d = await page.evaluate(() => {
                    const panel = document.getElementById('details-panel');
                    const pill = panel ? panel.querySelector('.readiness-pill') : null;
                    return {
                        selected: typeof currentNodeId !== 'undefined' ? currentNodeId : null,
                        pill: pill ? pill.textContent.trim() : null,
                        open: !!panel && !panel.classList.contains('hidden')
                    };
                });
                focusSelected = d.selected;
                focusPill = d.pill;
            }
            const fruitOk = th.panelVisible &&
                th.fruitRows === reportFruit && reportFruit > 0 &&
                th.fruitInState === reportFruit &&
                focusSelected === firstFruit.claim &&
                !!focusPill &&
                focusVisible !== null && focusVisible < beforeFocus.visibleNodes;

            // (b) gate edges drawn as gate edges, hubs synthesised.
            const gateOk = th.gateEdges >= graphGateEdges &&
                th.gateEdges > 0 &&
                th.gateStyled === th.gateEdges &&
                th.hubCount > 0 &&
                th.nodeCount === evidenceCount + th.hubCount;

            const rendersOk = th.renderMode === 'threads' && th.reportLoaded &&
                th.canvases >= 1 && th.readinessChipsVisible &&
                th.answeredColour === '#C1443C' &&
                (th.byReadiness['answered-unrecorded'] || 0) > 0 &&
                (th.byReadiness['neutral'] || 0) < th.nodeCount;

            // (c) round trip out through dissolution and back to evidence.
            await page.selectOption('#graph-select', 'dissolution');
            await waitLoaded(page);
            await sleep(900);
            await page.selectOption('#graph-select', 'evidence');
            await waitLoaded(page);
            await sleep(1200);
            const ev = await readEvidenceContent(firstFruit.claim);

            // and evidence is still *interactive*, not merely painted correctly.
            await typeSearch(page, target.id);
            await clickNode(page, target.id);
            const evDetail = await page.evaluate(() => {
                const panel = document.getElementById('details-panel');
                const pill = panel ? panel.querySelector('.status-pill') : null;
                return {
                    open: !!panel && !panel.classList.contains('hidden'),
                    pill: pill ? pill.textContent.trim() : null,
                    selected: typeof currentNodeId !== 'undefined' ? currentNodeId : null
                };
            });
            await typeSearch(page, '');

            // The evidence style the gate edges must be wearing is the project's
            // own, read from the same config the viewer reads -- not hardcoded here.
            const visualConfig = JSON.parse(
                await fsp.readFile(path.join(ROOT, '_data', 'visual_config.json'), 'utf8'));
            const configGate = ((visualConfig.edge_styles || {})['blocks']) || {};
            const configGateColour = String(configGate.color || '').toUpperCase();
            const configGateDashes = configGate.dashes;
            const evidenceGateStyleOk = ev.gateStyleSample.length > 0 &&
                ev.gateStyleSample.every(g => g.drawn &&
                    g.colour === configGateColour &&
                    g.dashes === JSON.stringify(configGateDashes));

            const roundTripOk = ev.renderMode === 'evidence' &&
                ev.nodeCount === evidenceCount &&
                ev.hubNodes === 0 &&
                ev.gateEdges === graphGateEdges &&
                ev.gateWearingThreadsStyle === 0 &&
                evidenceGateStyleOk &&
                ev.statusChecked > 0 && ev.statusWrongCount === 0 &&
                !!ev.probeWant && ev.probePaint === ev.probeWant &&
                !ev.readinessChipsVisible && !ev.fruitPanelVisible &&
                ev.threadsStateCleared &&
                evDetail.open && evDetail.selected === target.id && !!evDetail.pill;

            const ok14 = rendersOk && fruitOk && gateOk && roundTripOk && cleanSince(snap14);
            record(14, THREADS_TITLE,
                ok14,
                `threads: mode=${th.renderMode} overlay=${th.reportLoaded ? 'loaded' : 'MISSING'} ` +
                `nodes=${th.nodeCount} (evidence ${evidenceCount} + ${th.hubCount} synthesised hubs) ` +
                `edges=${th.edgeCount}${rendersOk ? '' : ' <-- RENDER'}\n` +
                `readiness: ` +
                Object.keys(th.byReadiness).sort().map(k => `${k}=${th.byReadiness[k]}`).join(' ') +
                `; answered-unrecorded painted ${th.answeredColour} (expected #C1443C)\n` +
                `fruit list: ${th.fruitRows} rows vs ${reportFruit} in threads.json; ` +
                `click rank 1 -> currentNodeId=${focusSelected} (report says ${firstFruit.claim}), ` +
                `pill=${focusPill === null ? 'MISSING' : '"' + focusPill + '"'}, ` +
                `visible ${beforeFocus.visibleNodes} -> ${focusVisible}` +
                `${fruitOk ? '' : ' <-- FRUIT'}\n` +
                `gate edges: ${th.gateEdges} drawn (>= ${graphGateEdges} in entities.json), ` +
                `${th.gateStyled} styled dashed/width-4/#6B4C8A${gateOk ? '' : ' <-- GATE'}\n` +
                `back via dissolution: mode=${ev.renderMode} nodes=${ev.nodeCount}/${evidenceCount} ` +
                `hubsLeft=${ev.hubNodes} gateEdges=${ev.gateEdges} ` +
                `wearingThreadsStyle=${ev.gateWearingThreadsStyle} ` +
                `(drawn ${ev.gateStyleSample.map(g => g.colour + '/' + g.dashes).join(', ')}; ` +
                `visual_config says ${configGateColour}/${JSON.stringify(configGateDashes)}) ` +
                `${ev.probeId} repainted ${ev.probePaint} (status ${ev.probeStatus} -> ${ev.probeWant}) ` +
                `statusColours ${ev.statusChecked - ev.statusWrongCount}/${ev.statusChecked} correct ` +
                `chips/panel hidden=${!ev.readinessChipsVisible && !ev.fruitPanelVisible} ` +
                `stateCleared=${ev.threadsStateCleared}${roundTripOk ? '' : ' <-- LEAKED'}\n` +
                (ev.statusWrongCount
                    ? 'mis-coloured: ' + ev.statusWrong.map(w =>
                        `${w.id} status=${w.status} want=${w.want} got=${w.got}`).join('; ') + '\n'
                    : '') +
                `evidence still interactive: details open=${evDetail.open} ` +
                `node=${evDetail.selected} status pill=` +
                `${evDetail.pill === null ? 'MISSING' : '"' + evDetail.pill + '"'}\n` +
                (describeSince(snap14).join('\n') || 'no errors'));
        }

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
        (results.length < TOTAL_ASSERTIONS
            ? `  (only ${results.length}/${TOTAL_ASSERTIONS} assertions ran)` : ''));
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
    if (failed.length || results.length < TOTAL_ASSERTIONS) exitCode = 1;
    process.exit(exitCode);
}

main();
