#!/usr/bin/env node
/**
 * Headless browser smoke + interaction test for docs/graph/viewer.html.
 *
 * Serves docs/graph on a free port, drives the viewer with Playwright's
 * bundled Chromium, and asserts eighteen things:
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
 *  15. the evidence graph's INDEX view is a deterministic ordered index: two
 *      independent loads in separate browser contexts, each asked for the
 *      index through the switch, put every node on the same coordinates,
 *      nothing drifts once it is drawn (physics is off in the only sense the
 *      reader can perceive — every node pinned, zero displacement over a
 *      settle window), and a round trip out to another graph and back restores
 *      those coordinates EXACTLY, not approximately
 *  16. that ordered index is LEGIBLE, not merely ordered: captions render at
 *      or above ORDERED_TARGET_LABEL_PX once the grid is fitted, the lane and
 *      date-band axes are actually named, and those headings do not swallow
 *      clicks meant for the graph underneath them
 *  17. the NETWORK/INDEX view switch is real, reversible and remembered: a
 *      first-time visitor arrives in Network with physics running, nothing
 *      pinned and every type in play; Index pins all of it; both round trips
 *      are exact in both directions; and the choice survives a fresh load
 *  18. the switch leaves the reader looking at the graph, and "Reset View"
 *      neither changes which view they are in nor stores one for them --
 *      17 checks the state the switch reaches, not the camera it leaves
 *      behind, and both of those shipped green underneath it
 *
 * Assertions 1-11, 13 and 14 run at 1600x1000. Assertion 12 opens a second,
 * phone-sized context (393x830, isMobile + hasTouch) so the narrow-screen
 * layout is pinned without disturbing the desktop measurements the others
 * depend on. Assertion 15 opens two further desktop contexts, because "the
 * same every load" is only testable across genuinely separate loads,
 * assertion 16 opens one more at 1600x1000 (caption size is a function of the
 * fitted zoom, so it has to be read on a window whose size is known), and
 * assertion 17 opens one more still, because a first-time visitor is by
 * definition a context with empty localStorage.
 *
 * THE DEFAULT VIEW MOVED. viewer.html's DEFAULT_VIEW is VIEW_NETWORK: the
 * evidence graph opens force-directed, and the ordered index is one tap away
 * rather than the thing that arrives. Assertions 15 and 16 were written when
 * the index was the arrival, so they now ASK for it first -- through the same
 * chip a reader clicks, not by poking the flag, so a switch that stopped
 * working would take them down with it. Neither budget moved: both are still
 * tolerance-zero, and both still measure the index and nothing else.
 *
 * Assertion 13 and assertion 15 are deliberately complementary, and both are
 * load-bearing. 13 pins the mode flags on the way back from another graph
 * (hierarchical off, physics option on, override cleared) and tolerates
 * generous positional drift, because it was written against a stochastic
 * force-directed arrival. 15 pins the thing 13 cannot see: that the arrival is
 * a pure function of the data, to the integer. Weakening 15 to "approximately
 * the same" would silently re-admit the physics cloud 13 already tolerates.
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
const TOTAL_ASSERTIONS = 18;

// --- ordered arrival budget (assertion 15) ---------------------------------
// The index property is "the same bytes in, the same pixels out". These are
// exact-match budgets on purpose: one unit of tolerance is one unit of physics,
// and the whole point of the ordered arrival is that there is none. The settle
// window is the interval over which an unpinned solver would visibly move
// things (the force-directed arrival this replaced was still travelling
// hundreds of units at t=3s).
const ORDERED_SETTLE_WINDOW_MS = 2500;
const ORDERED_MAX_DRIFT_UNITS = 0;   // total |dx|+|dy| across every node
const ORDERED_MAX_RELOAD_DELTA = 0;  // nodes allowed to differ between loads

// --- legibility budget (assertion 16) --------------------------------------
// Assertion 15 pins that the arrival is ordered and deterministic. It says
// nothing about whether a reader can READ it, and those are genuinely
// independent: reverting the grid pitch to the round numbers it started with
// re-creates the original 4.94px-label bug -- an unreadable index -- and 15
// still passes, because every node is still exactly where it deterministically
// ought to be. That gap was measured, not theorised.
//
// So this is the budget that stops the readability work silently rotting. The
// number is the same one viewer.html solves its pitch for
// (ORDERED_TARGET_LABEL_PX), read the same way: node font size * getScale(),
// with the claims-only arrival showing. Measured at LEGIBILITY_VIEWPORT
// because label size is height-bound -- a narrower window legitimately yields
// less, which is a known and documented limitation, not a regression.
const LEGIBILITY_VIEWPORT = { width: 1600, height: 1000 };
const MIN_EFFECTIVE_LABEL_PX = 11;
const MIN_AXIS_HEADINGS = 2;   // at least one lane and one band named

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
// Filled in by main() once the CDN mirror has been populated on disk.
let mirror = {};

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

// Attach the three collectors every page needs, together, so a page cannot be
// half-instrumented.
//
// This exists because a page WAS half-instrumented. Assertion 18 opens its own
// contexts to check framing and Reset View; the reset pages were given a
// pageerror listener, or none at all, and never a response listener. Since 18
// ends in `cleanSince(snap18)`, that meant a thrown error or a 404 raised
// during exactly the checks 18 was added for would have gone unrecorded and
// left it green -- the same "assertion that cannot fail" shape 15, 16 and 18
// were each written to close, reappearing inside one of them.
//
// Every listener reads the module-level `phase` when it FIRES, not when it is
// attached, so attributions follow whatever phase is current.
function instrumentPage(page) {
    page.on('pageerror', (e) => {
        pageErrors.push({ phase, text: (e && e.stack) || String(e) });
    });
    page.on('console', (msg) => {
        if (msg.type() === 'error') consoleErrors.push({ phase, text: msg.text() });
    });
    page.on('response', (res) => {
        const rec = { phase, status: res.status(), url: res.url() };
        allResponses.push(rec);
        if (rec.status >= 400) badResponses.push(rec);
    });
    return page;
}

// Serve the two CDN scripts from the local mirror. Paired with instrumentPage
// above: a page that routes but is not instrumented is the bug this fixes.
async function mirrorCdn(page) {
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
    return page;
}

// One call that cannot produce a partially-wired page.
async function newInstrumentedPage(ctx) {
    const page = await ctx.newPage();
    instrumentPage(page);
    await mirrorCdn(page);
    return page;
}

// The evidence graph opens in NETWORK view (viewer.html's DEFAULT_VIEW).
// Anything that wants to measure the ORDERED INDEX has to ask for it first --
// and asks the way a reader does, by clicking the switch in #chip-bar, so that
// a switch which silently stopped working takes the index assertions down with
// it instead of leaving them measuring a physics cloud and calling it a grid.
// Returns what it did, so the assertion can report "no #view-index" rather than
// timing out somewhere further along.
async function enterIndexView(page) {
    const outcome = await page.evaluate(() => {
        const el = document.getElementById('view-index');
        if (!el) return 'no #view-index in the DOM';
        const group = document.getElementById('view-switch');
        if (group && group.classList.contains('mode-hidden')) return 'switch hidden';
        if (el.classList.contains('view-active')) return 'already in index view';
        el.click();
        return 'clicked';
    });
    await sleep(900);
    await settle(page);
    return outcome;
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

    // Assigned, not declared: `mirror` is bound at module scope so the page
    // helpers defined above main() can serve from the same map.
    mirror = Object.fromEntries(
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

            // "answered-unrecorded" is the record-has-fallen-behind class, and
            // an empty one is the healthy state: it emptied on 2026-07-31 when
            // the operator's #54 ruling recorded H4's disposition and detector
            // A learned to retire a claim once a disposition exists. So the
            // paint check is conditional on membership: when the report says
            // the class is empty, demanding a painted node would fail the
            // build for being caught up. Readiness colouring in general is
            // still exercised: some non-neutral class must be populated.
            const answeredCount = th.byReadiness['answered-unrecorded'] || 0;
            const answeredOk = answeredCount > 0
                ? th.answeredColour === '#C1443C'
                : th.answeredColour === null;
            const rendersOk = th.renderMode === 'threads' && th.reportLoaded &&
                th.canvases >= 1 && th.readinessChipsVisible &&
                answeredOk &&
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
                (answeredCount > 0
                    ? `; answered-unrecorded painted ${th.answeredColour} (expected #C1443C)\n`
                    : `; answered-unrecorded class empty (record caught up; no paint expected)\n`) +
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

        // ---------------------------------------------------------------- 15
        // The evidence graph's INDEX view is a deterministic ordered index.
        //
        // The index is no longer what arrives -- viewer.html's DEFAULT_VIEW is
        // VIEW_NETWORK, on the owner's instruction -- so this asks for it
        // first, by clicking #view-index the way a reader does. That is the
        // only change: the property being measured, and the zero tolerance it
        // is measured to, are exactly as they were. Asking through the DOM
        // rather than through setViewMode() is deliberate; a switch that
        // stopped being wired up would fail here rather than quietly leaving
        // this assertion measuring a physics cloud and calling it a grid.
        //
        // This is the single property the ordered index rests on, and
        // it is the one thing none of assertions 1-14 can see. Assertion 13
        // checks the mode FLAGS after a round trip and tolerates half the
        // distinct-x count drifting away, because it was written when the
        // arrival was a physics cloud. If the ordered layout silently stopped
        // being applied — a stale `fixed`, an early return in
        // applyOrderedLayout(), a solver left running — 13 would still pass
        // while the front door went back to being a different picture every
        // load. So 15 measures the property directly, in three parts:
        //
        //   a. TWO LOADS, TWO CONTEXTS, IDENTICAL COORDINATES. Separate browser
        //      contexts, so nothing is shared: no cache, no storage, no
        //      surviving vis instance. Every node must land on exactly the same
        //      integer pair. Comparing within one page would prove nothing —
        //      it is the second *load* that has to agree.
        //
        //   b. NOTHING MOVES ONCE IT IS DRAWN. "Physics off" is asserted as the
        //      reader experiences it rather than as an options flag, because
        //      the viewer deliberately leaves physics.enabled true (assertion
        //      13 requires it) and instead pins every node. Both halves are
        //      checked: every node carries fixed.x && fixed.y, AND total
        //      displacement over a settle window is zero. Either alone is
        //      forgeable — a pinned node whose coordinates are recomputed on a
        //      timer would pass the first, and a solver that happens to be at
        //      rest on this data would pass the second.
        //
        //   c. THE ROUND TRIP RESTORES IT EXACTLY. Out to dissolution — the
        //      hierarchical override that has leaked before — and back. Not
        //      "roughly the same shape": the same integers. This is the
        //      layout-leak bug's second incarnation, and an ordered layout that
        //      comes back 3 units off is an ordered layout that is being
        //      recomputed from something other than the data.
        // ------------------------------------------------------------------
        phase = 'ordered-arrival';
        const snap15 = snapshotErrors();

        // Read every node's coordinates as integers, plus what the layout says
        // about itself. Rounding is the caller's problem nowhere: vis stores
        // these as floats and the ordered layout writes Math.round()ed ints, so
        // an exact comparison is legitimate here and would not be if physics
        // had ever touched them.
        const readOrdered = (p) => p.evaluate(() => {
            const pos = network.getPositions();
            const ids = Object.keys(pos).sort();
            const items = nodesDataSet.get();
            const pinned = items.filter(n => n.fixed && n.fixed.x === true && n.fixed.y === true).length;
            return {
                graphKey: typeof currentGraphKey !== 'undefined' ? currentGraphKey : null,
                orderedLayoutActive: typeof orderedLayoutActive !== 'undefined'
                    ? !!orderedLayoutActive : null,
                exploreMode: typeof exploreMode !== 'undefined' ? !!exploreMode : null,
                overrideCleared: typeof layoutOverride === 'undefined' || layoutOverride === null,
                hierarchical: !!(network.layoutEngine && network.layoutEngine.options &&
                    network.layoutEngine.options.hierarchical &&
                    network.layoutEngine.options.hierarchical.enabled),
                physicsOption: !!network.physics.options.enabled,
                nodeCount: ids.length,
                totalNodes: items.length,
                pinnedNodes: pinned,
                distinctX: new Set(ids.map(id => Math.round(pos[id].x))).size,
                coords: ids.map(id => `${id}:${Math.round(pos[id].x)},${Math.round(pos[id].y)}`)
            };
        });

        // Total displacement between two coordinate readings, plus the worst
        // offenders by name — a count alone does not tell you whether one node
        // slipped or the whole grid re-flowed.
        const displacement = (a, b) => {
            const parse = (rows) => {
                const m = new Map();
                rows.forEach(r => {
                    const i = r.lastIndexOf(':');
                    const [x, y] = r.slice(i + 1).split(',').map(Number);
                    m.set(r.slice(0, i), [x, y]);
                });
                return m;
            };
            const ma = parse(a), mb = parse(b);
            let total = 0;
            const movers = [];
            const missing = [];
            for (const [id, pa] of ma) {
                const pb = mb.get(id);
                if (!pb) { missing.push(id); continue; }
                const d = Math.abs(pa[0] - pb[0]) + Math.abs(pa[1] - pb[1]);
                if (d) { total += d; movers.push({ id, from: pa, to: pb, d }); }
            }
            for (const id of mb.keys()) if (!ma.has(id)) missing.push(id);
            movers.sort((x, y) => y.d - x.d);
            return {
                total, movedCount: movers.length, missingCount: missing.length,
                worst: movers.slice(0, 4).map(m =>
                    `${m.id} (${m.from[0]},${m.from[1]})->(${m.to[0]},${m.to[1]}) d=${m.d}`),
                missing: missing.slice(0, 4)
            };
        };

        let arrivalA = null, arrivalB = null, afterDrift = null, afterTrip = null;
        let driftDelta = null, reloadDelta = null, tripDelta = null;
        const orderedNotes = [];
        const enterNotes = [];

        for (const which of ['A', 'B']) {
            const oCtx = await browser.newContext({ viewport: { width: 1600, height: 1000 } });
            try {
                const oPage = await oCtx.newPage();
                await oPage.route('**/*', async (route) => {
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
                oPage.on('pageerror', (err) => {
                    pageErrors.push({ phase, text: (err && err.stack) || String(err) });
                });
                oPage.on('console', (msg) => {
                    if (msg.type() === 'error') consoleErrors.push({ phase, text: msg.text() });
                });
                oPage.on('response', (res) => {
                    const rec = { phase, status: res.status(), url: res.url() };
                    allResponses.push(rec);
                    if (rec.status >= 400) badResponses.push(rec);
                });

                await oPage.goto(`${base}/viewer.html`, { waitUntil: 'domcontentloaded' });
                await waitLoaded(oPage);
                // A fresh context has empty localStorage, so both loads open in
                // Network. Ask for the index through the switch. Recorded per
                // context because "the same every load" has to mean the same
                // gesture every load too.
                enterNotes.push(`${which}: ${await enterIndexView(oPage)}`);

                if (which === 'A') {
                    arrivalA = await readOrdered(oPage);
                    // (b) nothing moves. Deliberately NOT preceded by another
                    // settle() call: this window starts the moment the arrival
                    // is declared done, which is exactly when a reader is
                    // looking at it.
                    await sleep(ORDERED_SETTLE_WINDOW_MS);
                    afterDrift = await readOrdered(oPage);
                    driftDelta = displacement(arrivalA.coords, afterDrift.coords);

                    // (c) out through the hierarchical override and back.
                    await oPage.selectOption('#graph-select', 'dissolution');
                    await waitLoaded(oPage);
                    await sleep(1200);
                    const viaDiss = await readOrdered(oPage);
                    orderedNotes.push(
                        `detour dissolution: hierarchical=${viaDiss.hierarchical} ` +
                        `physicsOption=${viaDiss.physicsOption} ordered=${viaDiss.orderedLayoutActive}`);
                    await oPage.selectOption('#graph-select', 'evidence');
                    await waitLoaded(oPage);
                    await sleep(1500);
                    // Deliberately NOT re-clicking the switch here. The reader
                    // chose Index before the detour, and the graph they left in
                    // Index is the graph they must come back to; a viewer that
                    // needs the switch pressed again after every graph change
                    // has a preference it does not honour.
                    afterTrip = await readOrdered(oPage);
                    tripDelta = displacement(arrivalA.coords, afterTrip.coords);
                } else {
                    arrivalB = await readOrdered(oPage);
                    reloadDelta = displacement(arrivalA.coords, arrivalB.coords);
                }
            } finally {
                await oCtx.close().catch(() => {});
            }
        }
        phase = 'ordered-arrival';

        const shapeOk = !!arrivalA && arrivalA.graphKey === 'evidence' &&
            arrivalA.orderedLayoutActive === true && arrivalA.exploreMode === false &&
            arrivalA.overrideCleared && !arrivalA.hierarchical &&
            // Every node pinned — not "most", and not merely the visible ones:
            // an unpinned hidden node is a node that will have moved by the
            // time a type chip reveals it.
            arrivalA.pinnedNodes === arrivalA.totalNodes && arrivalA.totalNodes > 0 &&
            // An ordered grid has lanes. One distinct x is a stack, and one per
            // node is a cloud; the arrival is neither.
            arrivalA.distinctX > 1 && arrivalA.distinctX < arrivalA.nodeCount;

        const determinismOk = !!reloadDelta &&
            reloadDelta.missingCount === 0 &&
            reloadDelta.movedCount <= ORDERED_MAX_RELOAD_DELTA &&
            reloadDelta.total <= ORDERED_MAX_RELOAD_DELTA;
        const stillOk = !!driftDelta && driftDelta.total <= ORDERED_MAX_DRIFT_UNITS;
        const tripOk = !!afterTrip && !!tripDelta &&
            tripDelta.missingCount === 0 && tripDelta.total <= ORDERED_MAX_DRIFT_UNITS &&
            !afterTrip.hierarchical && afterTrip.physicsOption &&
            afterTrip.overrideCleared && afterTrip.orderedLayoutActive === true &&
            afterTrip.pinnedNodes === afterTrip.totalNodes;

        const ok15 = shapeOk && determinismOk && stillOk && tripOk && cleanSince(snap15);
        record(15,
            'the evidence graph\'s Index view is a deterministic ordered index ' +
            '(identical coordinates across two loads, nothing moving on arrival, ' +
            'restored exactly after a round trip)',
            ok15,
            (arrivalA
                ? `arrival: graph=${arrivalA.graphKey} ordered=${arrivalA.orderedLayoutActive} ` +
                  `explore=${arrivalA.exploreMode} override=${arrivalA.overrideCleared ? 'null' : 'SET'} ` +
                  `hierarchical=${arrivalA.hierarchical} ` +
                  `pinned=${arrivalA.pinnedNodes}/${arrivalA.totalNodes} ` +
                  `placed=${arrivalA.nodeCount} distinctX=${arrivalA.distinctX}` +
                  `${shapeOk ? '' : ' <-- NOT AN ORDERED ARRIVAL'}\n`
                : 'arrival: NOT MEASURED\n') +
            (driftDelta
                ? `physics off on arrival: over ${ORDERED_SETTLE_WINDOW_MS}ms ` +
                  `${driftDelta.movedCount} nodes moved, total drift ${driftDelta.total} units ` +
                  `(budget ${ORDERED_MAX_DRIFT_UNITS})${stillOk ? '' : ' <-- STILL SETTLING'}` +
                  `${driftDelta.worst.length ? '\n  ' + driftDelta.worst.join('\n  ') : ''}\n`
                : 'physics off on arrival: NOT MEASURED\n') +
            (reloadDelta
                ? `second load in a fresh context: ${arrivalB.nodeCount} nodes placed, ` +
                  `${reloadDelta.movedCount} differ, ${reloadDelta.missingCount} missing, ` +
                  `total delta ${reloadDelta.total} units (budget ${ORDERED_MAX_RELOAD_DELTA})` +
                  `${determinismOk ? '' : ' <-- NOT DETERMINISTIC'}` +
                  `${reloadDelta.worst.length ? '\n  ' + reloadDelta.worst.join('\n  ') : ''}` +
                  `${reloadDelta.missing.length ? '\n  missing: ' + reloadDelta.missing.join(', ') : ''}\n`
                : 'second load: NOT MEASURED\n') +
            `entered the index by clicking the switch: ${enterNotes.join('; ') || 'NOT ATTEMPTED'}\n` +
            orderedNotes.map(n => n + '\n').join('') +
            (tripDelta
                ? `back from dissolution: ${tripDelta.movedCount} nodes differ, ` +
                  `total delta ${tripDelta.total} units, ` +
                  `hierarchical=${afterTrip.hierarchical} physicsOption=${afterTrip.physicsOption} ` +
                  `override=${afterTrip.overrideCleared ? 'null' : 'SET'} ` +
                  `ordered=${afterTrip.orderedLayoutActive} ` +
                  `pinned=${afterTrip.pinnedNodes}/${afterTrip.totalNodes}` +
                  `${tripOk ? '' : ' <-- ORDERED LAYOUT LEAKED OR LOST'}` +
                  `${tripDelta.worst.length ? '\n  ' + tripDelta.worst.join('\n  ') : ''}\n`
                : 'round trip: NOT MEASURED\n') +
            (describeSince(snap15).join('\n') || 'no errors'));

        // ------------------------------------------------------------------
        // 16. The ordered index is legible, not merely ordered.
        //
        // An index is exhaustive, ordered, fast to scan and points elsewhere.
        // Assertion 15 covers ordered. This covers the scanning: captions big
        // enough to read, and axis headings naming what the lanes and bands
        // are -- because a grid whose axes are anonymous is a pretty pattern,
        // not an index.
        //
        // Both halves are load-bearing and neither implies the other. The
        // pitch constants in viewer.html are derived from label metrics; put
        // the original round numbers back and the whole grid inflates ~1.8x,
        // fit() zooms out to contain it, captions land at ~5px -- and 15 still
        // passes, every node still exactly where it belongs. Symmetrically the
        // headings are an HTML overlay that could be dropped without moving a
        // single node.
        // ------------------------------------------------------------------
        phase = 'legibility';
        const snap16 = snapshotErrors();

        let legibility = null;
        let legibilityEntry = 'not attempted';
        {
            const lCtx = await browser.newContext({ viewport: LEGIBILITY_VIEWPORT });
            try {
                const lPage = await lCtx.newPage();
                lPage.on('pageerror', (e) => pageErrors.push({ phase, text: String(e) }));
                lPage.on('console', (msg) => {
                    if (msg.type() === 'error') consoleErrors.push({ phase, text: msg.text() });
                });
                lPage.on('response', (res) => {
                    const rec = { phase, status: res.status(), url: res.url() };
                    allResponses.push(rec);
                    if (rec.status >= 400) badResponses.push(rec);
                });
                await lPage.route('**/*', async (route) => {
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
                await lPage.goto(`${base}/viewer.html`, { waitUntil: 'domcontentloaded' });
                await waitLoaded(lPage);
                // Same reason as assertion 15: this measures the INDEX, and the
                // index is no longer what arrives.
                legibilityEntry = await enterIndexView(lPage);
                await sleep(1200);

                legibility = await lPage.evaluate(() => {
                    const scale = network.getScale();
                    // The font the index actually draws claims at. Read from the
                    // viewer's own constant rather than re-derived here, so the
                    // test cannot drift away from the thing it is policing.
                    const font = (typeof ORDERED_FONT === 'number') ? ORDERED_FONT : 14;
                    const drawn = network.body.data.nodes.get()
                        .filter(n => n.hidden !== true)
                        .map(n => (n.label || '').trim())
                        .filter(Boolean);
                    const overlay = document.getElementById('axis-overlay');
                    const labels = overlay
                        ? Array.from(overlay.querySelectorAll('.axis-label'))
                            .filter(el => el.offsetWidth > 0 && (el.textContent || '').trim())
                        : [];
                    // A heading that swallows a click is worse than no heading:
                    // it makes a node under it unreachable.
                    let clickThrough = true;
                    for (const el of labels) {
                        const r = el.getBoundingClientRect();
                        const hit = document.elementFromPoint(
                            Math.round(r.left + r.width / 2), Math.round(r.top + r.height / 2));
                        if (hit && hit.tagName !== 'CANVAS') { clickThrough = false; break; }
                    }
                    return {
                        // `typeof`, not `window.orderedLayout`: the viewer
                        // declares it as a top-level `let`, which creates a
                        // declarative binding rather than a property of window.
                        // Reading it through window yields undefined always, so
                        // the transposed check silently answered "false" even on
                        // a phone where the layout really is transposed -- an
                        // assertion that cannot fail. Measured, not guessed.
                        transposed: (typeof orderedLayout !== 'undefined' && orderedLayout)
                            ? !!orderedLayout.transposed
                            : null,
                        scale: +scale.toFixed(3),
                        font,
                        effectivePx: +(font * scale).toFixed(2),
                        drawnLabels: drawn.length,
                        laneHeadings: labels.filter(el => el.classList.contains('axis-lane')).length,
                        bandHeadings: labels.filter(el => el.classList.contains('axis-band')).length,
                        headings: labels.length,
                        clickThrough
                    };
                });
            } finally {
                await lCtx.close().catch(() => {});
            }
        }

        const ok16 = !!legibility &&
            // A transposed (narrow) arrival draws no captions by design, so the
            // caption budget would be meaningless. This viewport is a desktop.
            legibility.transposed === false &&
            legibility.effectivePx >= MIN_EFFECTIVE_LABEL_PX &&
            legibility.drawnLabels > 0 &&
            legibility.laneHeadings >= 1 && legibility.bandHeadings >= 1 &&
            legibility.headings >= MIN_AXIS_HEADINGS &&
            legibility.clickThrough &&
            cleanSince(snap16);
        record(16,
            'the ordered index is legible: captions at or above the target size ' +
            'and named axes that do not swallow clicks',
            ok16,
            `entered the index by clicking the switch: ${legibilityEntry}\n` +
            (legibility
                ? `at ${LEGIBILITY_VIEWPORT.width}x${LEGIBILITY_VIEWPORT.height}: ` +
                  `fit scale ${legibility.scale}, font ${legibility.font} -> ` +
                  `effective label ${legibility.effectivePx}px ` +
                  `(budget >= ${MIN_EFFECTIVE_LABEL_PX})` +
                  `${legibility.effectivePx >= MIN_EFFECTIVE_LABEL_PX ? '' : ' <-- TOO SMALL TO READ'}` +
                  // Early warning, on a PASS. Caption size is font x the zoom
                  // that fits the grid, so it shrinks every time a claim is
                  // added to the record -- registering the 94th cost 0.72px.
                  // Without this line the first person to add a claim sees a
                  // viewer assertion go red for an edit to FINDINGS.md and has
                  // no way to connect the two.
                  `${legibility.effectivePx >= MIN_EFFECTIVE_LABEL_PX &&
                     legibility.effectivePx - MIN_EFFECTIVE_LABEL_PX < 1
                        ? `\n  NOTE: only ${(legibility.effectivePx - MIN_EFFECTIVE_LABEL_PX).toFixed(2)}px above the floor. ` +
                          `Captions shrink as the record grows (~0.7px per claim), so the next claim ` +
                          `registered in docs/FINDINGS.md is likely to fail this assertion. That would ` +
                          `be the grid outgrowing the viewport, not a viewer regression -- the fix is to ` +
                          `stop fitting the whole grid, not to lower the floor.`
                        : ''}\n` +
                  `captions drawn: ${legibility.drawnLabels}` +
                  `${legibility.drawnLabels ? '' : ' <-- NONE DRAWN'}` +
                  `${legibility.transposed ? ' <-- TRANSPOSED, expected the wide index' : ''}\n` +
                  `axis headings: ${legibility.laneHeadings} lane + ` +
                  `${legibility.bandHeadings} band = ${legibility.headings}` +
                  `${legibility.headings >= MIN_AXIS_HEADINGS &&
                     legibility.laneHeadings >= 1 && legibility.bandHeadings >= 1
                        ? '' : ' <-- AXES UNNAMED'}, ` +
                  `click-through ${legibility.clickThrough}` +
                  `${legibility.clickThrough ? '' : ' <-- HEADINGS EAT CLICKS'}`
                : 'NOT MEASURED') +
            '\n' + (describeSince(snap16).join('\n') || 'no errors'));

        // ---------------------------------------------------------------- 17
        // The view switch: two co-equal readings, reversibly.
        //
        // The index replaced the graph instead of joining it. It is now one of
        // two views, and NETWORK -- force-directed, physics running, nothing
        // pinned, every type in play -- is what a first-time visitor gets.
        // That makes three separate things breakable, and none of 1-16 can see
        // any of them:
        //
        //   a. THE DEFAULT. A context with empty localStorage must land in
        //      Network, and "Network" has to mean it: physics option on, ZERO
        //      pinned nodes, and the full type set. Not the 93-claim index
        //      default drawn as a cloud -- that is the index's node set wearing
        //      the graph's layout, which is neither view, and it is exactly
        //      what happens if the claims-only default is left keyed to the
        //      graph instead of to the view.
        //
        //   b. BOTH ROUND TRIPS, BOTH DIRECTIONS. This file has been bitten
        //      twice by a return path that half-worked (see the comment above
        //      `layoutOverride = null`), so each direction is measured on its
        //      own terms rather than assumed from the other:
        //        index -> network -> index  restores the SAME INTEGERS. Zero
        //          tolerance, for the same reason assertion 15 has none: one
        //          unit of drift is one unit of physics.
        //        network -> index -> network leaves physics GENUINELY RUNNING.
        //          Not "the option is true" -- the option is true in the index
        //          too, by design. Nodes have to actually move, which is the
        //          only reading a stale `fixed` cannot forge.
        //
        //   c. THE PREFERENCE PERSISTS. Checked by reloading the page, in both
        //      directions, because a preference that is only remembered in a
        //      JS variable is not remembered at all. Both reloads matter: one
        //      that only ever stores 'index' would also pass a viewer that had
        //      simply gone back to opening on the index.
        //
        // Assertions 15 and 16 lean on this: both now enter the index by
        // clicking #view-index, so a broken switch fails three assertions
        // rather than silently redefining what two of them measure.
        // ------------------------------------------------------------------
        phase = 'view-switch';
        const snap17 = snapshotErrors();

        // Everything the switch can get wrong, read off the live network.
        // `unpinnedNodes` counts `fixed === false` specifically rather than
        // "not pinned": vis leaves `fixed` undefined on a node nobody ever
        // pinned, and undefined is how a node that was never released looks.
        const readView = (p) => p.evaluate(() => {
            const pos = network.getPositions();
            const ids = Object.keys(pos).sort();
            const items = nodesDataSet.get();
            const types = (typeof activeTypes !== 'undefined' && activeTypes) ? activeTypes : {};
            const typeKeys = Object.keys(types);
            const group = document.getElementById('view-switch');
            // Key read from the page, not hardcoded here, for the same reason
            // assertion 16 reads ORDERED_FONT off the viewer: a rename in
            // viewer.html would otherwise leave 17 and 18 reading a key nobody
            // writes, seeing `stored === null` forever, and passing.
            let stored;
            const storageKey = (typeof VIEW_STORAGE_KEY === 'string')
                ? VIEW_STORAGE_KEY : 'lucier-graph-view';
            try { stored = window.localStorage.getItem(storageKey); }
            catch (e) { stored = 'THREW'; }
            return {
                view: (typeof currentView === 'function') ? currentView() : null,
                explore: (typeof exploreMode !== 'undefined') ? !!exploreMode : null,
                stored,
                ordered: (typeof orderedLayoutActive !== 'undefined') ? !!orderedLayoutActive : null,
                overrideCleared: typeof layoutOverride === 'undefined' || layoutOverride === null,
                hierarchical: !!(network.layoutEngine && network.layoutEngine.options &&
                    network.layoutEngine.options.hierarchical &&
                    network.layoutEngine.options.hierarchical.enabled),
                physicsOption: !!network.physics.options.enabled,
                totalNodes: items.length,
                pinnedNodes: items.filter(n => n.fixed && n.fixed.x === true && n.fixed.y === true).length,
                unpinnedNodes: items.filter(n => n.fixed === false).length,
                visibleNodes: items.filter(n => n.hidden !== true).length,
                typesOn: typeKeys.filter(k => types[k] !== false).length,
                typesTotal: typeKeys.length,
                switchShown: !!(group && !group.classList.contains('mode-hidden')),
                lit: ['view-network', 'view-index']
                    .filter(id => {
                        const el = document.getElementById(id);
                        return el && el.classList.contains('view-active');
                    }).join(',') || 'none',
                distinctX: new Set(ids.map(id => Math.round(pos[id].x))).size,
                coords: ids.map(id => `${id}:${Math.round(pos[id].x)},${Math.round(pos[id].y)}`)
            };
        });

        const clickView = async (p, id) => {
            const outcome = await p.evaluate((which) => {
                const el = document.getElementById(which);
                if (!el) return `no #${which}`;
                const group = document.getElementById('view-switch');
                if (group && group.classList.contains('mode-hidden')) return 'switch hidden';
                const rect = el.getBoundingClientRect();
                if (rect.width < 40 || rect.height < 40) {
                    // Not fatal on a desktop -- the 40x40 floor is the mobile
                    // rule -- but worth carrying into the report.
                    el.click();
                    return `clicked (${Math.round(rect.width)}x${Math.round(rect.height)})`;
                }
                el.click();
                return 'clicked';
            }, id);
            await sleep(900);
            return outcome;
        };

        let vArrival = null, vIndex1 = null, vNetwork1 = null, vIndex2 = null;
        let vMoved = null, vReloadIndex = null, vReloadNetwork = null;
        let indexTripDelta = null;
        const viewNotes = [];

        {
            const vCtx = await browser.newContext({ viewport: { width: 1600, height: 1000 } });
            try {
                const vPage = await vCtx.newPage();
                vPage.on('pageerror', (e) => pageErrors.push({ phase, text: (e && e.stack) || String(e) }));
                vPage.on('console', (msg) => {
                    if (msg.type() === 'error') consoleErrors.push({ phase, text: msg.text() });
                });
                vPage.on('response', (res) => {
                    const rec = { phase, status: res.status(), url: res.url() };
                    allResponses.push(rec);
                    if (rec.status >= 400) badResponses.push(rec);
                });
                await vPage.route('**/*', async (route) => {
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

                // (a) first-time visitor: nothing stored, nothing remembered.
                await vPage.goto(`${base}/viewer.html`, { waitUntil: 'domcontentloaded' });
                await waitLoaded(vPage);
                await sleep(1200);
                vArrival = await readView(vPage);

                // -> index
                viewNotes.push(`click #view-index: ${await clickView(vPage, 'view-index')}`);
                await settle(vPage);
                vIndex1 = await readView(vPage);

                // -> network. Sampled immediately and again a beat later: the
                // question is whether the solver is integrating, and the only
                // honest answer is that the nodes are somewhere else now.
                viewNotes.push(`click #view-network: ${await clickView(vPage, 'view-network')}`);
                const beforeMove = await readView(vPage);
                await sleep(1400);
                vNetwork1 = await readView(vPage);
                vMoved = displacement(beforeMove.coords, vNetwork1.coords);

                // -> index again. Same integers, or the return path is
                // recomputing the grid from something other than the data.
                viewNotes.push(`click #view-index: ${await clickView(vPage, 'view-index')}`);
                await settle(vPage);
                vIndex2 = await readView(vPage);
                indexTripDelta = displacement(vIndex1.coords, vIndex2.coords);

                // (c) the preference survives a real load, both ways round.
                await vPage.reload({ waitUntil: 'domcontentloaded' });
                await waitLoaded(vPage);
                await sleep(1200);
                vReloadIndex = await readView(vPage);

                viewNotes.push(`click #view-network: ${await clickView(vPage, 'view-network')}`);
                await vPage.reload({ waitUntil: 'domcontentloaded' });
                await waitLoaded(vPage);
                await sleep(1200);
                vReloadNetwork = await readView(vPage);
            } finally {
                await vCtx.close().catch(() => {});
            }
        }
        phase = 'view-switch';

        const defaultOk = !!vArrival &&
            vArrival.view === 'network' && vArrival.explore === true &&
            vArrival.ordered === true && vArrival.switchShown === true &&
            vArrival.lit === 'view-network' &&
            vArrival.physicsOption === true && !vArrival.hierarchical &&
            vArrival.overrideCleared &&
            vArrival.pinnedNodes === 0 &&
            vArrival.unpinnedNodes === vArrival.totalNodes &&
            // The full default node set, not the index's 93 claims.
            vArrival.visibleNodes === vArrival.totalNodes &&
            vArrival.typesOn === vArrival.typesTotal && vArrival.typesTotal > 0;

        const indexOk = !!vIndex1 &&
            vIndex1.view === 'index' && vIndex1.lit === 'view-index' &&
            vIndex1.pinnedNodes === vIndex1.totalNodes && vIndex1.totalNodes > 0 &&
            vIndex1.distinctX > 1 && vIndex1.distinctX < vIndex1.totalNodes &&
            vIndex1.visibleNodes < vIndex1.totalNodes;

        const backToNetworkOk = !!vNetwork1 && !!vMoved &&
            vNetwork1.view === 'network' && vNetwork1.lit === 'view-network' &&
            vNetwork1.physicsOption === true &&
            vNetwork1.pinnedNodes === 0 &&
            vNetwork1.unpinnedNodes === vNetwork1.totalNodes &&
            vNetwork1.visibleNodes === vNetwork1.totalNodes &&
            vNetwork1.typesOn === vNetwork1.typesTotal &&
            // Physics genuinely running, not merely enabled.
            vMoved.total > 0;

        const indexTripOk = !!indexTripDelta &&
            indexTripDelta.missingCount === 0 &&
            indexTripDelta.movedCount <= ORDERED_MAX_RELOAD_DELTA &&
            indexTripDelta.total <= ORDERED_MAX_RELOAD_DELTA &&
            !!vIndex2 && vIndex2.pinnedNodes === vIndex2.totalNodes;

        const persistOk = !!vReloadIndex && !!vReloadNetwork &&
            vReloadIndex.view === 'index' && vReloadIndex.stored === 'index' &&
            vReloadIndex.pinnedNodes === vReloadIndex.totalNodes &&
            vReloadNetwork.view === 'network' && vReloadNetwork.stored === 'network' &&
            vReloadNetwork.pinnedNodes === 0;

        const ok17 = defaultOk && indexOk && backToNetworkOk && indexTripOk &&
            persistOk && cleanSince(snap17);
        record(17,
            'the NETWORK/INDEX view switch is co-equal, reversible and remembered ' +
            '(Network is the default and really unpinned, both round trips are exact, ' +
            'the choice survives a reload)',
            ok17,
            (vArrival
                ? `first visit (empty storage): view=${vArrival.view} lit=${vArrival.lit} ` +
                  `stored=${vArrival.stored === null ? 'nothing' : vArrival.stored} ` +
                  `switchShown=${vArrival.switchShown} ` +
                  `pinned=${vArrival.pinnedNodes}/${vArrival.totalNodes} ` +
                  `released=${vArrival.unpinnedNodes}/${vArrival.totalNodes} ` +
                  `visible=${vArrival.visibleNodes}/${vArrival.totalNodes} ` +
                  `types=${vArrival.typesOn}/${vArrival.typesTotal} ` +
                  `physicsOption=${vArrival.physicsOption} distinctX=${vArrival.distinctX}` +
                  `${defaultOk ? '' : ' <-- NOT THE NETWORK VIEW'}\n`
                : 'first visit: NOT MEASURED\n') +
            (vIndex1
                ? `-> index: view=${vIndex1.view} lit=${vIndex1.lit} ` +
                  `pinned=${vIndex1.pinnedNodes}/${vIndex1.totalNodes} ` +
                  `visible=${vIndex1.visibleNodes}/${vIndex1.totalNodes} ` +
                  `distinctX=${vIndex1.distinctX}` +
                  `${indexOk ? '' : ' <-- NOT THE INDEX'}\n`
                : '-> index: NOT MEASURED\n') +
            (vNetwork1
                ? `-> network: view=${vNetwork1.view} lit=${vNetwork1.lit} ` +
                  `pinned=${vNetwork1.pinnedNodes}/${vNetwork1.totalNodes} ` +
                  `released=${vNetwork1.unpinnedNodes}/${vNetwork1.totalNodes} ` +
                  `visible=${vNetwork1.visibleNodes}/${vNetwork1.totalNodes} ` +
                  `types=${vNetwork1.typesOn}/${vNetwork1.typesTotal} ` +
                  `physicsOption=${vNetwork1.physicsOption}, ` +
                  `solver moved ${vMoved ? vMoved.movedCount : '?'} nodes ` +
                  `${vMoved ? vMoved.total : '?'} units in 1400ms` +
                  `${backToNetworkOk ? '' : ' <-- PHYSICS NOT ACTUALLY RUNNING'}\n`
                : '-> network: NOT MEASURED\n') +
            (indexTripDelta
                ? `index -> network -> index: ${indexTripDelta.movedCount} nodes differ, ` +
                  `total delta ${indexTripDelta.total} units (budget ${ORDERED_MAX_RELOAD_DELTA}), ` +
                  `pinned=${vIndex2.pinnedNodes}/${vIndex2.totalNodes}` +
                  `${indexTripOk ? '' : ' <-- ROUND TRIP NOT EXACT'}` +
                  `${indexTripDelta.worst.length ? '\n  ' + indexTripDelta.worst.join('\n  ') : ''}\n`
                : 'index round trip: NOT MEASURED\n') +
            (vReloadIndex && vReloadNetwork
                ? `after reload with 'index' remembered: view=${vReloadIndex.view} ` +
                  `stored=${vReloadIndex.stored} pinned=${vReloadIndex.pinnedNodes}/${vReloadIndex.totalNodes}\n` +
                  `after reload with 'network' remembered: view=${vReloadNetwork.view} ` +
                  `stored=${vReloadNetwork.stored} pinned=${vReloadNetwork.pinnedNodes}/${vReloadNetwork.totalNodes}` +
                  `${persistOk ? '' : ' <-- PREFERENCE NOT PERSISTED'}\n`
                : 'persistence: NOT MEASURED\n') +
            viewNotes.map(n => n + '\n').join('') +
            (describeSince(snap17).join('\n') || 'no errors'));

        // ---------------------------------------------------------------- 18
        // What the reader is actually LOOKING AT after using the switch.
        //
        // Assertion 17 proves the switch reaches the right state. It says
        // nothing about the camera, and nothing about the controls around it —
        // so both of these shipped with 17 green:
        //
        //   a. FRAMING. Index -> Network releases the pins and the solver
        //      expands the graph well past the index's extent, but the camera
        //      stayed on the index's zoom and pan. Measured before the fix:
        //      9.7% of nodes still on canvas at 1600x950, 13.1% at 393x830,
        //      and still 8.6% / 11.4% six seconds later — it never recovered.
        //      Every flag assertion 17 reads was correct the whole time.
        //
        //   b. RESET VIEW MUST NOT CHOOSE A VIEW. resetView() used to force the
        //      index and persist the choice, so one click moved a Network
        //      reader into the index permanently. It compounds with (a): when
        //      the graph flies off-screen, "Reset View" is the obvious
        //      recovery, so the natural sequence ended with the reader stuck in
        //      the view they never picked.
        //
        // The budget is deliberately loose. This failure is an order of
        // magnitude, not a few percent, and a tight threshold here would only
        // make the assertion flap on solver noise.
        phase = 'view-framing';
        const snap18 = snapshotErrors();
        const MIN_ONSCREEN_PCT = 60;

        let fArrival = null, fAfterSwitch = null, fReset = null, fResetReload = null;
        const frameNotes = [];

        {
            const fCtx = await browser.newContext({ viewport: { width: 1600, height: 1000 } });
            try {
                const fPage = await newInstrumentedPage(fCtx);

                // Share of drawn nodes whose canvas point lies inside the
                // container. Read through canvasToDOM so it is the camera being
                // measured, not the layout — a correct layout framed wrongly is
                // exactly the bug.
                const onScreen = (p) => p.evaluate(() => {
                    const c = network.body.container;
                    const ids = Object.keys(network.getPositions());
                    let inside = 0;
                    ids.forEach((id) => {
                        const pt = network.canvasToDOM(network.getPositions([id])[id]);
                        if (pt.x >= 0 && pt.x <= c.clientWidth &&
                            pt.y >= 0 && pt.y <= c.clientHeight) inside++;
                    });
                    return {
                        total: ids.length,
                        inside,
                        pct: ids.length ? +(100 * inside / ids.length).toFixed(1) : 0
                    };
                });

                await fPage.goto(`${base}/viewer.html`, { waitUntil: 'domcontentloaded' });
                await waitLoaded(fPage);
                await sleep(1400);
                fArrival = await onScreen(fPage);

                // (a) out to the index and back, by clicking, then let the
                // released solver run well past the settle the fix fits on.
                frameNotes.push(`click #view-index: ${await clickView(fPage, 'view-index')}`);
                await settle(fPage);
                frameNotes.push(`click #view-network: ${await clickView(fPage, 'view-network')}`);
                await sleep(5000);
                fAfterSwitch = await onScreen(fPage);

                // (b) Reset View, from Network, must leave the view alone AND
                // leave the stored preference alone.
                //
                // Checked on a page that has NOT touched the switch, which is
                // how the bug presented: a first-time visitor, nothing stored,
                // one click on Reset, and they are in the index for good. Doing
                // it on the page above would prove less — that page clicked its
                // way to Network, so a stored 'network' there is correct and
                // the assertion could not tell "reset wrote it" from "the click
                // wrote it".
                const rCtx = await browser.newContext({ viewport: { width: 1600, height: 1000 } });
                try {
                    const rPage = await newInstrumentedPage(rCtx);
                    await rPage.goto(`${base}/viewer.html`, { waitUntil: 'domcontentloaded' });
                    await waitLoaded(rPage);
                    await sleep(1200);
                    const rBefore = await readView(rPage);
                    frameNotes.push(`before Reset: view=${rBefore.view} ` +
                        `stored=${rBefore.stored === null ? 'nothing' : rBefore.stored}`);

                    await rPage.evaluate(() => { resetView(); });
                    await sleep(1600);
                    fReset = await readView(rPage);

                    // Same context, fresh page: if Reset persisted a choice,
                    // this is where it shows up. Reading it from the OTHER
                    // context would only report what the switch clicks stored.
                    const rPage2 = await newInstrumentedPage(rCtx);
                    await rPage2.goto(`${base}/viewer.html`, { waitUntil: 'domcontentloaded' });
                    await waitLoaded(rPage2);
                    await sleep(1200);
                    fResetReload = await readView(rPage2);
                } finally {
                    await rCtx.close().catch(() => {});
                }
            } catch (e) {
                frameNotes.push(`framing probe failed: ${(e && e.message) || e}`);
            } finally {
                await fCtx.close().catch(() => {});
            }
        }

        const framingOk = !!fArrival && !!fAfterSwitch &&
            fArrival.pct >= MIN_ONSCREEN_PCT && fAfterSwitch.pct >= MIN_ONSCREEN_PCT;
        // Reset must be a no-op on WHICH view, and must not write a preference
        // the reader never expressed.
        const resetOk = !!fReset && !!fResetReload &&
            fReset.view === 'network' && fReset.stored === null &&
            fResetReload.view === 'network' && fResetReload.stored === null;

        record(18,
            'using the switch leaves the graph on screen, and "Reset View" ' +
            'neither changes which view the reader is in nor remembers one for them',
            framingOk && resetOk && cleanSince(snap18),
            (fArrival && fAfterSwitch
                ? `nodes on canvas: arrival ${fArrival.inside}/${fArrival.total} (${fArrival.pct}%), ` +
                  `after index -> network ${fAfterSwitch.inside}/${fAfterSwitch.total} ` +
                  `(${fAfterSwitch.pct}%, budget >= ${MIN_ONSCREEN_PCT}%)` +
                  `${framingOk ? '' : ' <-- SWITCHED TO AN OFF-SCREEN GRAPH'}\n`
                : 'framing: NOT MEASURED\n') +
            (fReset
                ? `after Reset View from Network: view=${fReset.view} ` +
                  `stored=${fReset.stored === null ? 'nothing' : fReset.stored}` +
                  `${fReset.view === 'network' && fReset.stored === null ? '' : ' <-- RESET CHANGED THE VIEW'}\n`
                : 'reset: NOT MEASURED\n') +
            (fResetReload
                ? `reload afterwards: view=${fResetReload.view} ` +
                  `stored=${fResetReload.stored === null ? 'nothing' : fResetReload.stored}` +
                  `${fResetReload.view === 'network' ? '' : ' <-- RESET WAS PERSISTED'}\n`
                : 'reload after reset: NOT MEASURED\n') +
            frameNotes.map(n => n + '\n').join('') +
            (describeSince(snap18).join('\n') || 'no errors'));

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
