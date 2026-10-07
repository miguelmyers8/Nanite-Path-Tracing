/**
 * Path Trace Debugger: the Nanite cluster pipeline rasterized and path traced side by side, with the same cut.
 */

import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TeapotGeometry } from 'three/addons/geometries/TeapotGeometry.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

import { buildLodMeshletSetFromGeometry, MeshletMesh, MeshletScene, MeshletCullPass, MeshletColorMode, MaterialTable, LightingModel } from 'nanite/meshlets/index.js';
import { PathTracePass, PathTraceView, traceRay, cameraRay } from 'nanite-path-tracing/index.js';

const $ = ( id ) => document.getElementById( id );
const ui = {};
for ( const id of [ 'pathBadge', 'layout', 'threshold', 'thresholdVal', 'buildStats', 'bounces', 'bouncesVal', 'spp', 'sppVal', 'scale', 'scaleVal', 'split', 'splitVal', 'sunSize', 'sunSizeVal',
	'exposure', 'exposureVal', 'cullBack', 'reset', 'verify', 'viewMode', 'sun', 'sunVal', 'sunElevation', 'sunElevationVal', 'roughness', 'roughnessVal', 'metalness', 'metalnessVal',
	'gpuStats', 'fps', 'samplesTag', 'status', 'error', 'view' ] ) ui[ id ] = $( id );
const params = new URLSearchParams( location.search );
/** ?test: render offscreen (a headless WebGPU has no swap chain), run a few frames, verify, publish window.__gpuTest */
const TEST = params.has( 'test' );

const showError = ( msg ) => { ui.error.hidden = false; ui.error.textContent = String( msg ); if ( TEST ) window.__gpuTest = { done: true, ok: false, error: String( msg ) }; };
window.addEventListener( 'error', ( e ) => showError( `Error: ${ e.message }` ) );
window.addEventListener( 'unhandledrejection', ( e ) => showError( `Unhandled: ${ e.reason?.stack || e.reason }` ) );
const setStatus = ( t ) => { ui.status.textContent = t; };
const fmt = ( n ) => Math.round( n ).toLocaleString( 'en-US' );

// --- renderer -----------------------------------------------------------------------------------

if ( ! navigator.gpu ) showError( 'This page needs WebGPU. Use Chrome or Edge.' );
const renderer = new THREE.WebGPURenderer( { antialias: false } );
renderer.setPixelRatio( Math.min( window.devicePixelRatio, 2 ) );
renderer.setClearColor( 0x0b0e12, 1 );
renderer.toneMapping = THREE.ACESFilmicToneMapping;
ui.view.prepend( renderer.domElement );

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 50, 1, 0.1, 400 );
camera.position.set( 0, 7, 22 );
const controls = new OrbitControls( camera, renderer.domElement );
controls.enableDamping = true;
controls.target.set( 0, 1, 0 );

// --- meshes and materials --------------------------------------------------------------------------

const MESHES = [
	{ name: 'Torus knot', make: () => new THREE.TorusKnotGeometry( 1, 0.34, 256, 48 ) },
	{ name: 'Teapot', make: () => new TeapotGeometry( 1.1, 16 ) },
	{ name: 'Sphere', make: () => new THREE.SphereGeometry( 1.2, 128, 96 ) },
	{ name: 'Rounded box', make: () => new RoundedBoxGeometry( 1.6, 1.6, 1.6, 12, 0.3 ) },
	{ name: 'Capsule', make: () => new THREE.CapsuleGeometry( 0.6, 1.2, 24, 64 ) },
	{ name: 'Floor', make: () => new THREE.BoxGeometry( 1, 1, 1, 24, 1, 24 ), floor: true },
];
const FLOOR = MESHES.length - 1;
const MATERIALS = [
	{ pattern: 'plaster', tint: '#d8d2c4', roughness: 0.85, metalness: 0 }, { pattern: 'checker', tint: '#f2b544', roughness: 0.4, metalness: 0 },
	{ pattern: 'bricks', tint: '#c9735a', roughness: 0.7, metalness: 0 }, { pattern: 'stripes', tint: '#4fb3d9', roughness: 0.25, metalness: 0.9 },
	{ pattern: 'dots', tint: '#9a5fd9', roughness: 0.5, metalness: 0.2 }, { pattern: 'noise', tint: '#5fc98a', roughness: 0.3, metalness: 0.6 },
	{ pattern: 'checker', tint: '#9aa3ad', roughness: 0.6, metalness: 0, uvScale: [ 24, 24 ] },
];

function makePattern( name, size = 256 ) {

	const c = document.createElement( 'canvas' ); c.width = c.height = size;
	const g = c.getContext( '2d' ), cell = size / 8;
	g.fillStyle = '#d8d8d8'; g.fillRect( 0, 0, size, size );
	if ( name === 'checker' ) { for ( let j = 0; j < 8; j ++ ) for ( let i = 0; i < 8; i ++ ) { g.fillStyle = ( i + j ) % 2 ? '#2b2f36' : '#e9e9e9'; g.fillRect( i * cell, j * cell, cell, cell ); } }
	else if ( name === 'bricks' ) {

		g.fillStyle = '#c9c2b8'; g.fillRect( 0, 0, size, size );
		const bw = size / 4, bh = size / 8;
		for ( let j = 0; j < 8; j ++ ) for ( let i = - 1; i < 5; i ++ ) { const x = i * bw + ( j % 2 ) * bw / 2; g.fillStyle = `hsl(${ 12 + ( ( i * 7 + j * 13 ) % 5 ) * 4 }, 55%, ${ 42 + ( ( i * 3 + j * 5 ) % 4 ) * 4 }%)`; g.fillRect( x + 3, j * bh + 3, bw - 6, bh - 6 ); }

	} else if ( name === 'stripes' ) { for ( let i = - 16; i < 16; i ++ ) { g.fillStyle = i % 2 ? '#f0f0f0' : '#20242b'; g.beginPath(); g.moveTo( i * cell, 0 ); g.lineTo( ( i + 1 ) * cell, 0 ); g.lineTo( ( i + 1 ) * cell + size, size ); g.lineTo( i * cell + size, size ); g.closePath(); g.fill(); } }
	else if ( name === 'dots' ) { g.fillStyle = '#f2f0ea'; g.fillRect( 0, 0, size, size ); g.fillStyle = '#1c2127'; for ( let j = 0; j < 8; j ++ ) for ( let i = 0; i < 8; i ++ ) { g.beginPath(); g.arc( ( i + 0.5 ) * cell, ( j + 0.5 ) * cell, cell * 0.28, 0, Math.PI * 2 ); g.fill(); } }
	else if ( name === 'noise' ) {

		const img = g.createImageData( size, size ); let seed = 1234;
		const rnd = () => ( seed = ( seed * 1664525 + 1013904223 ) >>> 0 ) / 4294967296;
		const n = 16, cells = Array.from( { length: n * n }, () => rnd() );
		for ( let y = 0; y < size; y ++ ) for ( let x = 0; x < size; x ++ ) {

			const fx = x / size * n, fy = y / size * n, x0 = Math.floor( fx ) % n, y0 = Math.floor( fy ) % n, x1 = ( x0 + 1 ) % n, y1 = ( y0 + 1 ) % n, tx = fx - Math.floor( fx ), ty = fy - Math.floor( fy );
			const v = ( cells[ y0 * n + x0 ] * ( 1 - tx ) + cells[ y0 * n + x1 ] * tx ) * ( 1 - ty ) + ( cells[ y1 * n + x0 ] * ( 1 - tx ) + cells[ y1 * n + x1 ] * tx ) * ty;
			const i = ( y * size + x ) * 4; img.data[ i ] = img.data[ i + 1 ] = img.data[ i + 2 ] = 90 + v * 150; img.data[ i + 3 ] = 255;

		}

		g.putImageData( img, 0, 0 );

	} else {

		const img = g.createImageData( size, size ); let seed = 77;
		const rnd = () => ( seed = ( seed * 1664525 + 1013904223 ) >>> 0 ) / 4294967296;
		for ( let i = 0; i < img.data.length; i += 4 ) { const v = 205 + rnd() * 40; img.data[ i ] = img.data[ i + 1 ] = img.data[ i + 2 ] = v; img.data[ i + 3 ] = 255; }
		g.putImageData( img, 0, 0 );

	}

	return c;

}

const hexToRgb = ( hex ) => [ 1, 3, 5 ].map( ( i ) => parseInt( hex.slice( i, i + 2 ), 16 ) / 255 );
const MAX_INSTANCES = 256;

const state = { sets: null, scene: null, mesh: null, cull: null, tracer: null, materials: null, target: null, frameCount: 0, verify: null, verifying: false, readback: null, lastReadback: 0, lastMs: 0, timeSum: 0, timeFrames: 0 };

function normalizeGeometry( g ) {

	g.computeBoundingSphere();
	const s = g.boundingSphere, k = 1 / Math.max( 1e-9, s.radius );
	g.translate( - s.center.x, - s.center.y, - s.center.z ); g.scale( k, k, k );
	g.computeBoundingSphere(); g.computeBoundingBox();
	return g;

}

// --- layouts ---------------------------------------------------------------------------------------------

const _m = new THREE.Matrix4(), _p = new THREE.Vector3(), _q = new THREE.Quaternion(), _s = new THREE.Vector3( 1, 1, 1 ), _e = new THREE.Euler();

function layoutInstances() {

	const sc = state.scene, layout = ui.layout.value;
	let seed = 4242; const rnd = () => ( seed = ( seed * 1664525 + 1013904223 ) >>> 0 ) / 4294967296;
	sc.clear();
	const add = ( meshIndex, x, y, z, yaw = 0, scale = 1, sy = scale, sz = scale ) => {

		_q.setFromEuler( _e.set( 0, yaw, 0 ) ); _s.set( scale, sy, sz ); _p.set( x, y, z );
		return sc.addInstance( meshIndex, _m.compose( _p, _q, _s ) );

	};

	const floor = ( size ) => add( FLOOR, 0, - 0.1, 0, 0, size, 0.2, size );
	const objects = ( cols, rows, spacing ) => {

		for ( let j = 0; j < rows; j ++ ) for ( let i = 0; i < cols; i ++ ) {

			const mesh = Math.floor( rnd() * FLOOR ), s = 0.7 + rnd() * 0.7;
			add( mesh, ( i - ( cols - 1 ) / 2 ) * spacing + ( rnd() - 0.5 ) * 0.6, s, ( j - ( rows - 1 ) / 2 ) * spacing + ( rnd() - 0.5 ) * 0.6, rnd() * Math.PI * 2, s );

		}

	};

	if ( layout === 'showcase' ) { floor( 40 ); objects( 7, 4, 3.2 ); }
	else if ( layout === 'field' ) { floor( 80 ); objects( 15, 10, 2.8 ); }
	else if ( layout === 'ring' ) { floor( 30 ); for ( let i = 0; i < 12; i ++ ) { const a = i / 12 * Math.PI * 2; add( i % FLOOR, Math.cos( a ) * 4.5, 1, Math.sin( a ) * 4.5, - a ); } }
	else { floor( 20 ); add( 0, 0, 1.2, 0, 0.4, 1.2 ); }

	// materials: by mesh, the floor gets the grey checker
	for ( let i = 0; i < sc.instanceCount; i ++ ) {

		const k = sc.instanceMesh[ i ];
		state.materials.setInstanceMaterial( i, k === FLOOR ? 6 : ( k + i ) % 6 );

	}

	state.verify = null; state.readback = null;
	if ( state.tracer ) state.tracer.reset();
	syncSceneStats();

}

function syncSceneStats() {

	const sc = state.scene, merged = sc.set, t = state.tracer;
	if ( ! merged ) return;
	setStatus( `${ fmt( merged.meshletCount ) } clusters in ${ MESHES.length } meshes · ${ sc.instanceCount } instances · ${ fmt( sc.pairCount ) } pairs` );
	const a = t ? t.accel : null;
	ui.buildStats.innerHTML = `
		<dt>Clusters (all levels)</dt><dd>${ fmt( merged.meshletCount ) } <span>· ${ merged.levels.length } levels</span></dd>
		<dt>Leaf triangles</dt><dd>${ fmt( merged.leafTriangleCount ) }</dd>
		<dt>Instances</dt><dd>${ sc.instanceCount } <span>of ${ MAX_INSTANCES }</span></dd>
		${ a ? `<dt>Hierarchy nodes</dt><dd>${ fmt( a.layout.hierarchyNodeCount ) } <span>· depth ${ a.maxHierarchyDepth }</span></dd>
		<dt>Cluster BVH nodes</dt><dd>${ fmt( a.layout.clusterBvhNodeCount ) } <span>· depth ${ a.maxClusterDepth }</span></dd>
		<dt>Acceleration data</dt><dd>${ ( a.vec4Count * 16 / 1048576 ).toFixed( 2 ) } MB <span>· built in ${ a.buildTimeMs.toFixed( 0 ) } ms</span></dd>` : '' }`;

}

// --- build -------------------------------------------------------------------------------------------------

async function rebuild() {

	ui.pathBadge.textContent = 'building'; ui.pathBadge.className = 'badge warn';
	setStatus( 'building clusters and LOD DAGs' );
	await new Promise( ( r ) => requestAnimationFrame( () => r() ) );
	disposeAll();

	state.materials = new MaterialTable( { size: 256, maxMaterials: 8, maxInstances: MAX_INSTANCES } );
	if ( params.has( 'nomips' ) ) { state.materials.texture.generateMipmaps = false; state.materials.texture.minFilter = THREE.LinearFilter; }
	MATERIALS.forEach( ( d, i ) => state.materials.setMaterial( i, { image: makePattern( d.pattern ), color: hexToRgb( d.tint ), roughness: d.roughness, metalness: d.metalness, uvScale: d.uvScale || [ 2, 2 ], name: d.pattern } ) );

	const sets = [];
	for ( const m of MESHES ) {

		const g = m.floor ? m.make() : normalizeGeometry( m.make() );
		const set = buildLodMeshletSetFromGeometry( g, { attributes: true, maxTriangles: 128, maxVertices: 128 } );
		set.name = m.name;
		sets.push( set );
		setStatus( `${ m.name }: ${ fmt( set.meshletCount ) } clusters, ${ set.levels.length } levels` );
		await new Promise( ( r ) => setTimeout( r, 0 ) );

	}

	state.sets = sets;
	state.scene = new MeshletScene( sets, { maxInstances: MAX_INSTANCES } );
	state.mesh = new MeshletMesh( state.scene.set, { scene: state.scene, materials: state.materials } );
	state.mesh.colorMode = MeshletColorMode.MATERIAL;
	state.mesh.lighting = LightingModel.PHYSICAL;
	state.mesh.lightTwoSided = false;
	state.mesh.smoothShading = true;
	scene.add( state.mesh );
	state.cull = new MeshletCullPass( state.mesh );
	layoutInstances();
	createTracer();
	applySettings();
	ui.pathBadge.textContent = 'WebGPU · path tracing'; ui.pathBadge.className = 'badge ok';

}

function disposeAll() {

	if ( state.tracer ) { state.tracer.dispose(); state.tracer = null; }
	if ( state.mesh ) { scene.remove( state.mesh ); state.mesh.dispose(); state.mesh = null; }
	if ( state.scene ) { state.scene.dispose(); state.scene = null; }
	if ( state.materials ) { state.materials.dispose(); state.materials = null; }

}

function frameSize() {

	const size = renderer.getDrawingBufferSize( new THREE.Vector2() );
	const s = parseFloat( ui.scale.value );
	return { w: Math.max( 1, size.x ), h: Math.max( 1, size.y ), fw: Math.max( 1, Math.round( size.x * s ) ), fh: Math.max( 1, Math.round( size.y * s ) ) };

}

function createTracer() {

	if ( ! state.mesh ) return;
	const { w, h, fw, fh } = frameSize();
	const storageBufferLimit = renderer.backend?.device?.limits?.maxStorageBuffersPerShaderStage ?? 8;
	if ( ! state.tracer ) state.tracer = new PathTracePass( state.mesh, fw, fh, { materials: state.materials, maxBounces: parseInt( ui.bounces.value, 10 ), storageBufferLimit } );
	else state.tracer.setSize( fw, fh );
	state.tracer.setDisplaySize( w, h );
	if ( TEST ) { if ( state.target ) state.target.dispose(); state.target = new THREE.RenderTarget( w, h ); }
	syncSceneStats();

}

// --- settings ----------------------------------------------------------------------------------------------------

function thresholdPx() { return Math.pow( 2, parseFloat( ui.threshold.value ) ); }
const VIEWS = { path: PathTraceView.PATH, albedo: PathTraceView.ALBEDO, normal: PathTraceView.NORMAL, cluster: PathTraceView.CLUSTER, level: PathTraceView.LEVEL, instance: PathTraceView.INSTANCE, triangle: PathTraceView.TRIANGLE, cost: PathTraceView.COST };
const RASTER_VIEWS = { path: MeshletColorMode.MATERIAL, albedo: MeshletColorMode.MATERIAL, normal: MeshletColorMode.NORMAL, cluster: MeshletColorMode.MESHLET, level: MeshletColorMode.LEVEL, instance: MeshletColorMode.INSTANCE, triangle: MeshletColorMode.MESHLET, cost: MeshletColorMode.MESHLET };

function applySettings() {

	ui.thresholdVal.textContent = `${ thresholdPx().toFixed( 1 ) } px`;
	ui.bouncesVal.textContent = ui.bounces.value; ui.sppVal.textContent = ui.spp.value;
	ui.scaleVal.textContent = parseFloat( ui.scale.value ).toFixed( 2 ); ui.splitVal.textContent = `${ ui.split.value } %`;
	ui.sunSizeVal.textContent = `${ parseFloat( ui.sunSize.value ).toFixed( 2 ) }°`; ui.exposureVal.textContent = parseFloat( ui.exposure.value ).toFixed( 2 );
	ui.sunVal.textContent = `${ ui.sun.value }°`; ui.sunElevationVal.textContent = `${ ui.sunElevation.value }°`;
	ui.roughnessVal.textContent = parseFloat( ui.roughness.value ).toFixed( 2 ); ui.metalnessVal.textContent = parseFloat( ui.metalness.value ).toFixed( 2 );
	const t = state.tracer, mesh = state.mesh;
	if ( ! t ) return;
	const view = ui.viewMode.value;
	t.view = VIEWS[ view ] ?? PathTraceView.PATH;
	t.maxBounces = parseInt( ui.bounces.value, 10 );
	t.samplesPerFrame = parseInt( ui.spp.value, 10 );
	t.sunAngularRadius = parseFloat( ui.sunSize.value ) * Math.PI / 180;
	t.exposure = parseFloat( ui.exposure.value );
	t.cullBackFaces = ui.cullBack.checked;
	t.lodThreshold = thresholdPx();
	state.cull.lodThreshold = thresholdPx();
	mesh.colorMode = RASTER_VIEWS[ view ] ?? MeshletColorMode.MATERIAL;
	mesh.lighting = view === 'path' ? LightingModel.PHYSICAL : view === 'albedo' ? LightingModel.UNLIT : LightingModel.SIMPLE;
	const a = parseFloat( ui.sun.value ) * Math.PI / 180, el = parseFloat( ui.sunElevation.value ) * Math.PI / 180;
	mesh.setLightDirection( Math.cos( el ) * Math.cos( a ), Math.sin( el ), Math.cos( el ) * Math.sin( a ) );   // the tracer shares the mesh's lighting uniforms
	const r = parseFloat( ui.roughness.value ) / 0.5, m = parseFloat( ui.metalness.value );
	MATERIALS.forEach( ( d, i ) => state.materials.setSurface( i, Math.min( 1, d.roughness * r ), Math.min( 1, Math.max( d.metalness, m ) ) ) );
	t.reset();

}

// --- verification: the GPU's primary hits against the CPU reference ------------------------------------------------

async function verify() {

	const t = state.tracer;
	if ( ! t || state.verifying ) return;
	state.verifying = true;
	t.jitter = false; t.reset();
	await new Promise( ( r ) => requestAnimationFrame( () => requestAnimationFrame( r ) ) ); // a frame traced through the pixel centres
	try {

		const W = t.width, H = t.height, WH = W * H;
		const data = new Float32Array( await renderer.getArrayBufferAsync( t.buffers.frame, null, WH * 16, WH * 16 ) );
		const ctx = t.getTraceContext(), lod = t.getLodParams();
		const cw = t.uniforms.cameraWorld.value.elements, fov = camera.fov * Math.PI / 180;
		const nx = Math.min( 64, W ), ny = Math.min( 48, H );
		let compared = 0, mismatches = 0, hits = 0, costSum = 0;
		for ( let p = 0; p < WH; p ++ ) costSum += data[ p * 4 + 3 ];
		for ( let j = 0; j < ny; j ++ ) for ( let i = 0; i < nx; i ++ ) {

			const px = Math.floor( ( i + 0.5 ) * W / nx ), py = Math.floor( ( j + 0.5 ) * H / ny ), o = ( py * W + px ) * 4;
			const gInst = data[ o ] - 1, gCluster = data[ o + 1 ], gTri = data[ o + 2 ];
			const { origin, direction } = cameraRay( cw, fov, W, H, px, py );
			const c = traceRay( ctx, origin, direction, lod );
			const same = c ? ( gInst === c.instance && gCluster === c.cluster && gTri === c.triangle ) : gInst < 0;
			compared ++; if ( ! same ) mismatches ++; if ( c ) hits ++;

		}

		state.verify = { compared, mismatches, hits, avgCost: costSum / WH, threshold: lod.threshold };

	} catch ( err ) { console.warn( 'verify failed', err ); } finally {

		t.jitter = true; t.reset(); state.verifying = false;

	}

}

function syncGpuStats() {

	const t = state.tracer, v = state.verify;
	if ( ! t ) return;
	ui.gpuStats.innerHTML = `
		<dt>Samples per pixel</dt><dd>${ fmt( t.sampleCount ) }</dd>
		<dt>Frame</dt><dd>${ state.lastMs.toFixed( 1 ) } ms <span>· ${ t.width }×${ t.height } · ${ t.maxBounces } bounces</span></dd>
		<dt>Primary hits GPU vs CPU</dt><dd class="${ v ? ( v.mismatches === 0 ? 'ok' : 'bad' ) : '' }">${ v ? ( v.mismatches === 0 ? `identical (${ fmt( v.compared ) } rays, ${ fmt( v.hits ) } hits)` : `${ fmt( v.mismatches ) } of ${ fmt( v.compared ) } differ` ) : '—' }</dd>
		<dt>Traversal cost per primary ray</dt><dd>${ v ? `${ v.avgCost.toFixed( 1 ) } <span>nodes + triangles</span>` : '—' }</dd>`;

}

// --- UI wiring -------------------------------------------------------------------------------------------------

ui.layout.addEventListener( 'change', () => { if ( state.scene ) { layoutInstances(); applySettings(); } } );
for ( const el of [ ui.viewMode, ui.cullBack ] ) el.addEventListener( 'change', applySettings );
for ( const el of [ ui.threshold, ui.bounces, ui.spp, ui.sunSize, ui.exposure, ui.sun, ui.sunElevation, ui.roughness, ui.metalness ] ) el.addEventListener( 'input', applySettings );
ui.split.addEventListener( 'input', () => { ui.splitVal.textContent = `${ ui.split.value } %`; } );
ui.scale.addEventListener( 'input', () => { ui.scaleVal.textContent = parseFloat( ui.scale.value ).toFixed( 2 ); createTracer(); applySettings(); } );
ui.reset.addEventListener( 'click', () => state.tracer && state.tracer.reset() );
ui.verify.addEventListener( 'click', verify );

function resize() {

	// ?test: a fixed small canvas (a software WebGPU traces a few pixels per millisecond)
	const w = TEST ? parseInt( params.get( 'w' ) || '160', 10 ) : ui.view.clientWidth, h = TEST ? parseInt( params.get( 'h' ) || '120', 10 ) : ui.view.clientHeight;
	if ( ! w || ! h ) return;
	renderer.setSize( w, h, false );
	camera.aspect = w / h; camera.updateProjectionMatrix();
	if ( state.tracer ) { createTracer(); applySettings(); }

}

new ResizeObserver( resize ).observe( ui.view );

// --- frame --------------------------------------------------------------------------------------------------------

let frames = 0, fpsTime = performance.now();

function animate( t ) {

	controls.update(); camera.updateMatrixWorld();
	if ( ! state.tracer ) { if ( ! TEST ) renderer.render( scene, camera ); return; }
	const t0 = performance.now();
	const { w, h } = frameSize();
	const tracer = state.tracer, cull = state.cull;
	const split = parseFloat( ui.split.value ) / 100 * w;
	tracer.split = split;
	// the raster frame (its own cull and indirect draw), then the path traced image over it, right of the split
	if ( TEST ) renderer.setRenderTarget( state.target );
	if ( split > 0 || TEST ) {

		cull.setViewport( camera, h / renderer.getPixelRatio() ); cull.execute( renderer, camera );
		renderer.render( scene, camera );

	} else renderer.clear();

	tracer.setViewport( camera, h );
	tracer.execute( renderer, camera );
	tracer.render( renderer );
	if ( TEST ) renderer.setRenderTarget( null );
	state.lastMs = performance.now() - t0;
	frames ++; state.frameCount ++;
	if ( t - fpsTime > 500 ) {

		ui.fps.textContent = `${ ( frames * 1000 / ( t - fpsTime ) ).toFixed( 0 ) } fps · ${ ( ( t - fpsTime ) / frames ).toFixed( 1 ) } ms · ${ tracer.width }×${ tracer.height }`;
		frames = 0; fpsTime = t; syncGpuStats();

	}

	ui.samplesTag.textContent = `${ fmt( tracer.sampleCount ) } spp`;

}

async function boot() {

	try {

		const adapter = await navigator.gpu.requestAdapter();
		if ( adapter && renderer.backend?.parameters ) renderer.backend.parameters.requiredLimits = { maxStorageBuffersPerShaderStage: Math.min( 16, adapter.limits.maxStorageBuffersPerShaderStage ) };

	} catch ( err ) { /* defaults */ }
	try { await renderer.init(); } catch ( err ) { showError( `WebGPU init failed: ${ err.message }. This page has no WebGL fallback.` ); return; }
	if ( ! renderer.backend?.isWebGPUBackend ) { showError( 'WebGPU is not available; this page has no WebGL fallback.' ); return; }
	if ( TEST ) { ui.bounces.value = params.get( 'bounces' ) || '1'; ui.layout.value = params.get( 'layout' ) || 'ring'; ui.split.value = params.get( 'split' ) || '50'; }
	resize();
	renderer.setAnimationLoop( animate );
	await rebuild();
	if ( TEST ) {

		// a few frames, then the CPU comparison, then the verdict
		await new Promise( ( r ) => setTimeout( r, 1500 ) );
		await verify();
		const v = state.verify;
		window.__gpuTest = { done: true, ok: !! v && v.mismatches <= Math.max( 2, v.compared * 0.002 ), verify: v, frames: state.frameCount, samples: state.tracer.sampleCount, clusters: state.scene.set.meshletCount, instances: state.scene.instanceCount, lastMs: state.lastMs };

	}

}

window.pathTraceDebugger = state; state.camera = camera; state.controls = controls; state.getRenderer = () => renderer; state.rebuild = rebuild; state.verify = verify;
boot();
