/**
 * Headless WebGPU harness: starts a static server over the repository, launches Chrome with
 * the flags that expose WebGPU through SwiftShader (no GPU needed), connects Playwright over
 * CDP, serves the pinned three.js CDN URLs from node_modules (no network needed), opens a page
 * and waits for it to publish `window.__gpuTest = { done: true, ... }`.
 *
 *   import { runGpuPage } from './gpu-harness.mjs';
 *   const result = await runGpuPage( '/test/gpu/tsl-smoke.html' );
 *
 * Needs a Chromium with WebGPU (Playwright's works) and the `playwright` package: either a
 * local devDependency or a global install (PLAYWRIGHT_MODULE / CHROME_PATH override the lookup).
 */
import { spawn } from 'node:child_process';
import http from 'node:http';
import { promises as fs, createReadStream, existsSync, openSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve( path.dirname( fileURLToPath( import.meta.url ) ), '..' );
const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8', '.json': 'application/json', '.css': 'text/css', '.wasm': 'application/wasm', '.png': 'image/png', '.jpg': 'image/jpeg', '.svg': 'image/svg+xml' };

async function loadPlaywright() {

	const candidates = [ process.env.PLAYWRIGHT_MODULE, 'playwright', '/opt/node-tools/node_modules/playwright/index.mjs', '/usr/lib/node_modules/playwright/index.mjs', '/usr/local/lib/node_modules/playwright/index.mjs' ].filter( Boolean );
	for ( const c of candidates ) { try { return await import( c ); } catch ( e ) { /* next */ } }
	throw new Error( 'gpu-harness: playwright not found; npm i -D playwright, or set PLAYWRIGHT_MODULE' );

}

function findChrome() {

	if ( process.env.CHROME_PATH ) return process.env.CHROME_PATH;
	const base = process.env.PLAYWRIGHT_BROWSERS_PATH || path.join( process.env.HOME || '', '.cache/ms-playwright' );
	if ( existsSync( base ) ) {

		const dirs = require_dirs( base ).filter( ( d ) => /^chromium-\d+$/.test( d ) ).sort();
		for ( const d of dirs.reverse() ) { const p = path.join( base, d, 'chrome-linux/chrome' ); if ( existsSync( p ) ) return p; }

	}

	for ( const p of [ '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser' ] ) if ( existsSync( p ) ) return p;
	throw new Error( 'gpu-harness: no Chromium found; set CHROME_PATH' );

}

function require_dirs( p ) { try { return require( 'node:fs' ).readdirSync( p ); } catch { return []; } }
import { createRequire } from 'node:module';
const require = createRequire( import.meta.url );

/** Static file server over the repository root. */
function serve( port ) {

	const server = http.createServer( async ( req, res ) => {

		try {

			let url = decodeURIComponent( new URL( req.url, 'http://x' ).pathname );
			if ( url.endsWith( '/' ) ) url += 'index.html';
			const file = path.join( root, url );
			if ( ! file.startsWith( root ) ) { res.writeHead( 403 ); res.end(); return; }
			const stat = await fs.stat( file ).catch( () => null );
			if ( ! stat || ! stat.isFile() ) { res.writeHead( 404 ); res.end( 'not found' ); return; }
			res.writeHead( 200, { 'Content-Type': MIME[ path.extname( file ) ] || 'application/octet-stream', 'Cache-Control': 'no-store' } );
			createReadStream( file ).pipe( res );

		} catch ( e ) { res.writeHead( 500 ); res.end( String( e ) ); }

	} );
	return new Promise( ( resolve ) => server.listen( port, '127.0.0.1', () => resolve( server ) ) );

}

/**
 * Run one page and return its `window.__gpuTest` once `done` is set (or the timeout).
 * @param {string} pagePath   path under the repository root, with an optional query string
 * @param {Object} [options]  { timeoutMs, port, debugPort, onConsole }
 */
export async function runGpuPage( pagePath, options = {} ) {

	const { timeoutMs = 120000, port = 8765 + Math.floor( Math.random() * 100 ), debugPort = 9400 + Math.floor( Math.random() * 100 ) } = options;
	const server = await serve( port );
	const chromePath = findChrome();
	const profile = path.join( process.env.TMPDIR || '/tmp', `gpu-harness-profile-${ process.pid }` );
	const chrome = spawn( chromePath, [
		'--headless=new', '--no-sandbox', '--disable-gpu-sandbox', '--ignore-gpu-blocklist',
		'--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader',
		'--window-size=1024,768', `--remote-debugging-port=${ debugPort }`, `--user-data-dir=${ profile }`, 'about:blank',
		...( process.env.GPU_HARNESS_LOG ? [ '--enable-logging=stderr', '--v=1', '--vmodule=*swiftshader*=2,*dawn*=2' ] : [] ),
	], { stdio: process.env.GPU_HARNESS_LOG ? [ 'ignore', 'ignore', openSync( process.env.GPU_HARNESS_LOG, 'a' ) ] : 'ignore' } );
	const { chromium } = await loadPlaywright();
	let browser = null;
	const logs = [];
	try {

		for ( let i = 0; i < 40 && ! browser; i ++ ) {

			await new Promise( ( r ) => setTimeout( r, 250 ) );
			browser = await chromium.connectOverCDP( `http://127.0.0.1:${ debugPort }` ).catch( () => null );

		}

		if ( ! browser ) throw new Error( 'gpu-harness: could not connect to Chrome' );
		const context = browser.contexts()[ 0 ] || await browser.newContext();
		const page = await context.newPage();
		await page.setViewportSize( { width: 1024, height: 768 } );
		// three r185 sets GPUTextureViewDescriptor.swizzle to the string form of the final spec; the Chromium builds Playwright
		// ships still validate it as a dictionary and throw. Drop the field (it is only meaningful with a feature we never enable).
		await page.addInitScript( () => {

			// keep every shader module's source and compilation messages, and surface uncaptured device errors
			window.__shaders = [];
			const origModule = GPUDevice.prototype.createShaderModule;
			GPUDevice.prototype.createShaderModule = function ( d ) {

				const module = origModule.call( this, d );
				const rec = { code: d.code, messages: [] };
				window.__shaders.push( rec );
				module.getCompilationInfo().then( ( info ) => { rec.messages = info.messages.map( ( m ) => `${ m.type} ${ m.lineNum }:${ m.linePos } ${ m.message }` ); } ).catch( () => {} );
				return module;

			};

			// pipeline creation errors (Tint validation at pipeline time, backend compile failures) with their messages
			const origCP = GPUDevice.prototype.createComputePipeline;
			GPUDevice.prototype.createComputePipeline = function ( d ) {

				this.pushErrorScope( 'validation' ); this.pushErrorScope( 'internal' );
				const pipeline = origCP.call( this, d );
				this.popErrorScope().then( ( e ) => { if ( e ) console.error( 'createComputePipeline internal error: ' + e.message ); } );
				this.popErrorScope().then( ( e ) => { if ( e ) console.error( 'createComputePipeline validation error: ' + e.message ); } );
				return pipeline;

			};

			const origCPA = GPUDevice.prototype.createComputePipelineAsync;
			GPUDevice.prototype.createComputePipelineAsync = function ( d ) { return origCPA.call( this, d ).catch( ( e ) => { console.error( 'createComputePipelineAsync: ' + e.message ); throw e; } ); };
			const origRP = GPUDevice.prototype.createRenderPipeline;
			GPUDevice.prototype.createRenderPipeline = function ( d ) {

				this.pushErrorScope( 'validation' );
				const pipeline = origRP.call( this, d );
				this.popErrorScope().then( ( e ) => { if ( e ) console.error( 'createRenderPipeline validation error: ' + e.message ); } );
				return pipeline;

			};

			const origDevice = GPUAdapter.prototype.requestDevice;
			GPUAdapter.prototype.requestDevice = async function ( d ) {

				const device = await origDevice.call( this, d );
				device.addEventListener( 'uncapturederror', ( e ) => console.error( 'WebGPU uncaptured error: ' + e.error.message ) );
				device.lost.then( ( info ) => console.error( `WebGPU device lost (${ info.reason }): ${ info.message }` ) );
				return device;

			};

			const orig = GPUTexture.prototype.createView;
			GPUTexture.prototype.createView = function ( d ) {

				if ( d && typeof d.swizzle === 'string' ) { const { swizzle, ...rest } = d; return orig.call( this, rest ); }
				return orig.call( this, d );

			};

		} );
		page.on( 'console', ( m ) => { logs.push( `[${ m.type() }] ${ m.text() }` ); if ( options.onConsole ) options.onConsole( m ); } );
		page.on( 'pageerror', ( e ) => logs.push( `[pageerror] ${ e.message }` ) );
		// the examples load three.js from jsDelivr: serve those URLs from node_modules so the test runs offline
		await page.route( /^https:\/\/cdn\.jsdelivr\.net\/npm\/three@[^/]+\/(.*)$/, async ( route ) => {

			const rel = route.request().url().match( /three@[^/]+\/(.*)$/ )[ 1 ].split( '?' )[ 0 ];
			const file = path.join( root, 'node_modules/three', rel );
			if ( existsSync( file ) ) route.fulfill( { path: file, contentType: MIME[ path.extname( file ) ] || 'text/javascript' } );
			else route.abort();

		} );
		await page.route( /^https:\/\/fonts\.(googleapis|gstatic)\.com\/.*/, ( route ) => route.fulfill( { status: 200, contentType: 'text/css', body: '' } ) );
		await page.goto( `http://127.0.0.1:${ port }${ pagePath }`, { waitUntil: 'load' } );
		const t0 = Date.now();
		let result = null;
		while ( Date.now() - t0 < timeoutMs ) {

			result = await page.evaluate( () => window.__gpuTest || null ).catch( () => null );
			if ( result && result.done ) break;
			await new Promise( ( r ) => setTimeout( r, 250 ) );

		}

		if ( options.screenshot ) await page.screenshot( { path: options.screenshot } );
		if ( result && typeof result.png === 'string' && result.png.startsWith( 'data:image/png;base64,' ) ) {

			const file = options.pngPath || path.join( root, 'test/.tmp', `${ path.basename( pagePath.split( '?' )[ 0 ], '.html' ) }.png` );
			await fs.mkdir( path.dirname( file ), { recursive: true } );
			await fs.writeFile( file, Buffer.from( result.png.slice( 22 ), 'base64' ) );
			result.png = file;

		}
		const shaders = await page.evaluate( () => ( window.__shaders || [] ).map( ( s ) => ( { code: s.code, messages: s.messages } ) ) ).catch( () => [] );
		const failed = ! ( result && result.done && result.ok !== false );
		if ( failed || options.dumpShaders ) {

			const dir = path.join( root, 'test/.tmp/shaders' );
			await fs.mkdir( dir, { recursive: true } );
			for ( let i = 0; i < shaders.length; i ++ ) await fs.writeFile( path.join( dir, `shader-${ i }.wgsl` ), shaders[ i ].code );
			for ( const s of shaders ) for ( const m of s.messages ) if ( ! /^info/.test( m ) ) logs.push( `[shader] ${ m }` );
			logs.push( `[harness] ${ shaders.length } shader modules written to test/.tmp/shaders/` );

		}

		return { result, logs, timedOut: ! ( result && result.done ) };

	} finally {

		if ( browser ) await browser.close().catch( () => {} );
		chrome.kill();
		server.close();
		await fs.rm( profile, { recursive: true, force: true } ).catch( () => {} );

	}

}
