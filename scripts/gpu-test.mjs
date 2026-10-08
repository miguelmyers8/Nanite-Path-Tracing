/**
 * Run the headless WebGPU tests: `node scripts/gpu-test.mjs [page ...]`
 * (default: every page listed below). Exit code 1 when any page reports `ok: false` or times out.
 */
import { runGpuPage } from './gpu-harness.mjs';

const pages = process.argv.slice( 2 ).length ? process.argv.slice( 2 ) : [ '/test/gpu/tsl-smoke.html', '/test/gpu/rand.html', '/test/gpu/pathtrace-parity.html', '/examples/pathtrace-debugger/?test=1&nomips=1' ];
let failed = 0;
for ( const p of pages ) {

	const t0 = Date.now();
	const { result, logs, timedOut } = await runGpuPage( p, { timeoutMs: 300000 } );
	const ok = ! timedOut && result && result.ok !== false;
	if ( ! ok ) failed ++;
	console.log( `${ ok ? 'PASS' : 'FAIL' } ${ p } (${ ( ( Date.now() - t0 ) / 1000 ).toFixed( 1 ) } s)` );
	console.log( JSON.stringify( result, null, 2 ) );
	if ( ! ok ) console.log( logs.slice( - 30 ).join( '\n' ) );

}

process.exit( failed ? 1 : 0 );
