/**
 * Real-time path tracing of a `MeshletMesh` (compute, TSL).
 *
 * One invocation per pixel per frame. The kernel builds the camera ray,
 * traces it through the instance TLAS, each hit instance's cluster hierarchy
 * (choosing the LOD cut on the way with the cull kernel's own rule, so the
 * traced surface is the rasterized surface) and the cluster's triangle BVH,
 * reading triangles through the mesh's corner fetch (the decode every draw
 * path uses). It shades the hit with the material table (or the flat
 * colour), a sun (next-event estimation through a shadow ray, with the
 * raster's light convention: a Lambert surface facing the sun returns
 * `albedo × lightColor`) and the hemisphere sky as the environment, samples
 * the next direction from a Lambert + GGX lobe pair, and accumulates the
 * radiance of `samplesPerFrame` paths into a per-pixel sum that the display
 * quad divides by the sample count. Moving the camera, resizing and changing
 * the pass's own settings reset the sum (real time at one sample per pixel,
 * converging when still); the pass cannot see the instance matrices, the light
 * or the materials change, so the caller calls `reset()` after changing them.
 *
 * Two kernels per bounce, wavefront style: the trace kernel traces the
 * path ray of every pixel (bounce 0 builds the camera ray) and records the
 * hit; the shade kernel shades it, traces the sun's shadow ray and samples
 * the next direction. The path state lives in the frame buffer between
 * them, so each kernel holds one traversal and stays small (short kernels
 * keep more rays in flight, and the split is where ray compaction or sorting
 * would go).
 *
 * The frame buffer is nine vec4 per pixel (`FRAME_BYTES_PER_PIXEL`), so a large frame needs the adapter's storage binding
 * limit in the device's `requiredLimits`; `maxPixels` makes the pass scale a larger request down instead of failing.
 *
 * Storage buffers (WebGPU's guaranteed eight): accel (hierarchies, cluster
 * BVHs, order, cluster table), tlas (instance nodes and records), frame
 * (colour sum per pixel; the primary hit's ids, cost and barycentrics for
 * the debug views and the parity check; the path state), and the mesh's
 * meta, vertices, triangles, vertexData.
 *
 * Views: PATH (accumulated), NORMAL, CLUSTER (the raster's cluster colours),
 * LEVEL, INSTANCE, COST (nodes + triangles tested by the primary ray),
 * ALBEDO, TRIANGLE: the display draws them from the primary hit record, so
 * the kernel only traces primary rays while one is shown.
 *
 * `traceRay()` in `src/core/trace.js` is the CPU twin of the traversal.
 *
 * @module PathTracePass
 */

import { Color, Vector2, Vector3, Matrix4, StorageBufferAttribute, QuadMesh, MeshBasicNodeMaterial } from 'three/webgpu';
import {
	Fn, If, Loop, Break, Discard, array, uniform, storage, instanceIndex, screenCoordinate, vec2, vec3, vec4, float, uint, int,
	select, normalize, cross, dot, length, max, min, abs, sqrt, pow, sin, cos, clamp, mix, floor, floatBitsToUint, uintBitsToFloat, hash, step,
} from 'three/tsl';
import { META_STRIDE_VEC4, META_OWN_SPHERE, META_PARENT_SPHERE, META_LOD_INFO } from 'nanite/meshlets/MeshletMesh.js';
import { simpleShade } from 'nanite/materials/Lighting.js';
import {
	buildAccel, buildInstanceTlas, meshBoxes, HIER_VEC4, CBVH_VEC4, TLAS_NODE_VEC4, TLAS_INSTANCE_VEC4, LEAF_BIT,
} from '../core/accel.js';
import { RAY_EPSILON, STACK_TLAS, STACK_HIERARCHY, STACK_CLUSTER } from '../core/trace.js';

/** Bytes per pixel of the frame buffer: nine vec4 (colour sum, primary hit record, barycentrics, six path-state slots). */
export const FRAME_BYTES_PER_PIXEL = 9 * 16;

/**
 * The most pixels one frame buffer can hold on a device: its storage binding and buffer size limits (WebGPU's defaults are
 * 128 MiB and 256 MiB, which is about 930 000 pixels; ask the adapter for more in the device's `requiredLimits`).
 * @param {GPUDevice} [device]  `renderer.backend.device`
 */
export function maxFramePixels( device ) {

	const l = device && device.limits;
	const bytes = Math.min( l ? l.maxStorageBufferBindingSize : 134217728, l ? l.maxBufferSize : 268435456 );
	return Math.max( 1, Math.floor( bytes / FRAME_BYTES_PER_PIXEL ) );

}

/**
 * One PCG step on a uint variable node: advances the state and returns the float in [0, 1) it produced.
 * The result is a variable assigned here, in statement order. A bare expression would be inlined where it is used, after any
 * later state update, and every number drawn between two updates would read the same final state (identical jitter in x and y,
 * a sun sample and a bounce direction on a one-dimensional curve): a test pins this (test/gpu/rand.html).
 * @param {Node} state  a `uint( seed ).toVar()`
 */
export function pcgRand( state ) {

	state.assign( state.mul( uint( 747796405 ) ).add( uint( 2891336453 ) ) );
	const word = state.shiftRight( state.shiftRight( uint( 28 ) ).add( uint( 4 ) ) ).bitXor( state ).mul( uint( 277803737 ) );
	return word.shiftRight( uint( 22 ) ).bitXor( word ).toFloat().mul( 1 / 4294967296 ).toVar();

}

export const PathTraceView = Object.freeze( { PATH: 0, NORMAL: 1, CLUSTER: 2, LEVEL: 3, INSTANCE: 4, COST: 5, ALBEDO: 6, TRIANGLE: 7 } );

const WORKGROUP_SIZE = 64;
const MAX_STEPS = 8192;
const _m = new Matrix4();

export class PathTracePass {

	/**
	 * @param {import('nanite/meshlets/MeshletMesh.js').MeshletMesh} mesh  an uncompressed mesh (plain or scene); its buffers are read as they are
	 * @param {number} width   pixels
	 * @param {number} height
	 * @param {Object} [options]
	 * @param {import('nanite/materials/MaterialTable.js').MaterialTable} [options.materials]  defaults to the mesh's table
	 * @param {Object} [options.lighting]   uniform nodes { lightDirection, lightColor, skyColor, groundColor, roughness, metalness, flatColor };
	 *                                      defaults to the mesh's own, so the raster and the tracer light the same way
	 * @param {number} [options.maxBounces=3]
	 * @param {number} [options.samplesPerFrame=1]
	 * @param {Object} [options.accel]      buildAccel options
	 * @param {number} [options.storageBufferLimit=8]
	 * @param {number} [options.workgroupSize=64]  invocations per workgroup of the trace kernel
	 * @param {number} [options.maxPixels]         the frame's pixel cap (maxFramePixels( renderer.backend.device )); default: none
	 */
	constructor( mesh, width, height, options = {} ) {

		if ( mesh.pages ) throw new Error( 'PathTracePass: paged (compressed) meshes are not supported yet' );
		if ( mesh.cameraRelative ) throw new Error( 'PathTracePass: camera-relative (chunked) meshes are not supported yet' );
		this.mesh = mesh;
		/** invocations per workgroup of the trace kernel (a software WebGPU such as SwiftShader needs a small one: its workgroup's private memory lives on one thread's stack) */
		this.workgroupSize = Math.max( 1, options.workgroupSize ?? WORKGROUP_SIZE );
		this.materials = options.materials || mesh.materials || null;
		this.lighting = options.lighting || mesh.uniforms;
		const limit = options.storageBufferLimit ?? 8;
		if ( this.storageBufferCount > limit ) throw new Error( `PathTracePass: the kernel binds ${ this.storageBufferCount } storage buffers, the device allows ${ limit }` );

		const set = mesh.meshletSet;
		/** the static acceleration structures of the set (core/accel.js) */
		this.accel = buildAccel( set, options.accel || {} );
		if ( this.accel.maxHierarchyDepth >= STACK_HIERARCHY - 1 || this.accel.maxClusterDepth >= STACK_CLUSTER - 1 ) throw new Error( `PathTracePass: hierarchy depth ${ this.accel.maxHierarchyDepth } / cluster depth ${ this.accel.maxClusterDepth } exceed the kernel's stacks` );
		this.buffers = { accel: new StorageBufferAttribute( this.accel.data, 4 ) };
		this.maxInstances = mesh.maxInstances;
		const tlasVec4 = 2 * this.maxInstances * TLAS_NODE_VEC4 + this.maxInstances * TLAS_INSTANCE_VEC4;
		this.buffers.tlas = new StorageBufferAttribute( new Float32Array( tlasVec4 * 4 ), 4 );
		this._tlas = null;
		this.meshBox = meshBoxes( set );
		this.meshes = set.meshes || [ { first: 0, count: set.meshletCount } ];

		this.uniforms = {
			width: uniform( 1, 'uint' ), height: uniform( 1, 'uint' ),
			cameraWorld: uniform( new Matrix4() ),
			cameraPosition: uniform( new Vector3() ),
			tanHalfFov: uniform( 0.5 ), aspect: uniform( 1 ),
			frame: uniform( 0, 'uint' ),
			sample: uniform( 0, 'uint' ),
			bounce: uniform( 0, 'uint' ),
			reset: uniform( 1, 'uint' ),
			jitter: uniform( 1, 'uint' ),
			samplesPerFrame: uniform( Math.max( 1, options.samplesPerFrame ?? 1 ), 'uint' ),
			maxBounces: uniform( Math.max( 0, options.maxBounces ?? 3 ), 'uint' ),
			view: uniform( PathTraceView.PATH, 'uint' ),
			/** LOD: the cull kernel's metric, from the camera for every ray */
			enableLod: uniform( set.lod ? 1 : 0, 'uint' ),
			lodPixelScale: uniform( 1000, 'float' ),
			lodThreshold: uniform( 1, 'float' ),
			lodNear: uniform( 0.01, 'float' ),
			forceLevel: uniform( - 1, 'float' ),
			/** sun: angular radius (radians) of the disk the shadow rays sample */
			sunAngularRadius: uniform( 0.5 * Math.PI / 180, 'float' ),
			cullBackFaces: uniform( 0, 'uint' ),
			tlasInstanceBase: uniform( 0, 'uint' ),
			tlasInstanceCount: uniform( 0, 'uint' ),
			/** display */
			exposure: uniform( 1, 'float' ),
			split: uniform( 0, 'float' ),      // viewport pixels from the left that show what was drawn underneath (the raster)
			viewport: uniform( new Vector2( 1, 1 ) ), // the drawing buffer the display covers (the frame may be smaller: a resolution scale)
			costMax: uniform( 400, 'float' ),
			background: uniform( new Vector3().fromArray( new Color( 0x0b0e12 ).toArray() ) ),   // the page's clear colour, in the linear working space
		};

		/** the frame never holds more pixels than this (maxFramePixels( device )): a larger request is scaled down, keeping its shape */
		this.maxPixels = Math.max( 1, options.maxPixels ?? Infinity );
		this._jitter = true;
		this._width = 0; this._height = 0; this.frameCount = 0; this.sampleCount = 0;
		this._needsReset = true;
		this.material = new MeshBasicNodeMaterial();
		this.material.depthTest = false; this.material.depthWrite = false;
		this.quad = new QuadMesh( this.material );
		this.setSize( width, height );

	}

	/** Storage buffers the trace kernel binds: accel, tlas, frame, meta, vertices, triangles, vertexData. */
	get storageBufferCount() { return 7; }

	set maxBounces( v ) { if ( ( v | 0 ) !== this.uniforms.maxBounces.value ) { this.uniforms.maxBounces.value = Math.max( 0, v | 0 ); this.reset(); } }
	get maxBounces() { return this.uniforms.maxBounces.value; }
	set samplesPerFrame( v ) { this.uniforms.samplesPerFrame.value = Math.max( 1, v | 0 ); }
	get samplesPerFrame() { return this.uniforms.samplesPerFrame.value; }
	set view( v ) { if ( ( v | 0 ) !== this.uniforms.view.value ) { this.uniforms.view.value = v | 0; this.reset(); } }
	get view() { return this.uniforms.view.value; }
	set lodThreshold( v ) { v = Math.max( 1e-6, v ); if ( v !== this.uniforms.lodThreshold.value ) { this.uniforms.lodThreshold.value = v; this.reset(); } }
	get lodThreshold() { return this.uniforms.lodThreshold.value; }
	set lodTest( v ) { const n = v ? 1 : 0; if ( n !== this.uniforms.enableLod.value ) { this.uniforms.enableLod.value = n; this.reset(); } }
	get lodTest() { return this.uniforms.enableLod.value === 1; }
	set forceLevel( v ) { if ( v !== this.uniforms.forceLevel.value ) { this.uniforms.forceLevel.value = v; this.reset(); } }
	get forceLevel() { return this.uniforms.forceLevel.value; }
	/** Random sub-pixel positions (the path view); the debug views always trace the pixel centre. */
	set jitter( v ) { this._jitter = !! v; }
	get jitter() { return this._jitter; }
	set sunAngularRadius( v ) { v = Math.max( 0, v ); if ( v !== this.uniforms.sunAngularRadius.value ) { this.uniforms.sunAngularRadius.value = v; this.reset(); } }
	get sunAngularRadius() { return this.uniforms.sunAngularRadius.value; }
	set cullBackFaces( v ) { const n = v ? 1 : 0; if ( n !== this.uniforms.cullBackFaces.value ) { this.uniforms.cullBackFaces.value = n; this.reset(); } }
	get cullBackFaces() { return this.uniforms.cullBackFaces.value === 1; }
	set exposure( v ) { this.uniforms.exposure.value = v; }
	get exposure() { return this.uniforms.exposure.value; }
	set split( v ) { this.uniforms.split.value = v; }
	get split() { return this.uniforms.split.value; }
	set costMax( v ) { this.uniforms.costMax.value = Math.max( 1, v ); }
	setBackground( r, g, b ) { this.uniforms.background.value.set( r, g, b ); }
	get costMax() { return this.uniforms.costMax.value; }
	get width() { return this._width; }
	get height() { return this._height; }

	/** The drawing buffer size the display quad covers; the frame is stretched over it when it is smaller (a resolution scale). */
	setDisplaySize( width, height ) { this.uniforms.viewport.value.set( Math.max( 1, width ), Math.max( 1, height ) ); }

	/** Start the accumulation over (the next frame overwrites the sums). */
	reset() { this._needsReset = true; }

	/** Pixel scale and near clamp from the camera and the viewport height in pixels, as MeshletCullPass.setViewport. */
	setViewport( camera, heightPixels ) {

		const fov = ( camera.fov ?? 50 ) * Math.PI / 180;
		const pixelScale = heightPixels / ( 2 * Math.tan( fov / 2 ) );
		if ( pixelScale !== this.uniforms.lodPixelScale.value ) this.reset();
		this.uniforms.lodPixelScale.value = pixelScale;
		this.uniforms.lodNear.value = Math.max( 1e-4, camera.near ?? 0.01 );

	}

	/** Resize the frame buffer (pixels); the accumulation restarts. */
	setSize( width, height ) {

		width = Math.max( 1, width | 0 ); height = Math.max( 1, height | 0 );
		if ( width * height > this.maxPixels ) {

			const k = Math.sqrt( this.maxPixels / ( width * height ) );
			width = Math.max( 1, Math.floor( width * k ) ); height = Math.max( 1, Math.floor( height * k ) );
			while ( width * height > this.maxPixels && width > 1 ) width --;

		}

		if ( width === this._width && height === this._height ) return false;
		this._width = width; this._height = height;
		this.uniforms.width.value = width; this.uniforms.height.value = height;
		if ( this.uniforms.viewport.value.x <= 1 ) this.uniforms.viewport.value.set( width, height );
		const n = width * height;
		if ( this.buffers.frame ) this.buffers.frame.dispose?.();
		/** per pixel: [r, g, b, samples] sums; [instance + 1, cluster, triangle, cost] and [u, v, t, 0] of the frame's first primary ray;
		 * then the path state between the kernels: ray origin + done, direction + rng, throughput, radiance, hit (t, ids), hit (u, v, cost) */
		this.buffers.frame = new StorageBufferAttribute( new Float32Array( n * 9 * 4 ), 4 );
		this.frameNode = storage( this.buffers.frame, 'vec4', n * 9 );
		this.pixelCount = n;
		// the size is compiled into the kernels: release the pipelines of the ones this replaces
		if ( this.traceKernel ) this.traceKernel.dispose();
		if ( this.shadeKernel ) this.shadeKernel.dispose();
		this._createKernels();
		this.reset();
		return true;

	}

	/** The LOD inputs the GPU used, for the CPU reference (`traceRay`'s `lod` argument). */
	getLodParams() {

		const u = this.uniforms;
		return { cameraPosition: u.cameraPosition.value.toArray(), pixelScale: u.lodPixelScale.value, threshold: u.lodThreshold.value, near: u.lodNear.value, forceLevel: u.forceLevel.value, lodTest: u.enableLod.value === 1 };

	}

	/** The traversal context of the CPU reference: { set, accel, tlas } for the frame last traced. */
	getTraceContext() { return { set: this.mesh.meshletSet, accel: this.accel, tlas: this._tlas }; }

	/**
	 * Rebuild the TLAS from the mesh's instances (CPU) and upload it. Called by `execute()`; a caller that keeps the
	 * instances still can skip it with `execute( renderer, camera, { tlas: false } )` after the first frame.
	 */
	updateTlas() {

		const mesh = this.mesh, scene = mesh.scene;
		mesh.updateWorldMatrix( true, false );
		const matrices = scene ? scene.instanceMatrices.array : mesh.buffers.instanceMatrices.array;
		this._tlas = buildInstanceTlas( {
			instanceCount: mesh.instanceCount, instanceMatrices: matrices, modelMatrix: mesh.matrixWorld.elements,
			instanceMesh: scene ? scene.instanceMesh : null, meshes: this.meshes, meshBox: this.meshBox, meshRoot: this.accel.meshRoot,
		}, this.buffers.tlas.array );
		const t = this._tlas;
		if ( t.data !== this.buffers.tlas.array ) throw new Error( 'PathTracePass: more instances than the mesh capacity' );
		this.buffers.tlas.addUpdateRange( 0, t.vec4Count * 4 );
		this.buffers.tlas.needsUpdate = true;
		this.uniforms.tlasInstanceBase.value = t.instanceBase;
		this.uniforms.tlasInstanceCount.value = t.instanceCount;

	}

	/**
	 * Trace one frame: refresh the camera and the TLAS, run the kernel. Draw the result with `render()`.
	 * @param {Object} [options]
	 * @param {boolean} [options.tlas=true]  rebuild the TLAS from the instances
	 */
	execute( renderer, camera, options = {} ) {

		camera.updateMatrixWorld();
		const u = this.uniforms;
		const path = u.view.value === PathTraceView.PATH;
		_m.copy( camera.matrixWorld );
		if ( ! _m.equals( u.cameraWorld.value ) ) { u.cameraWorld.value.copy( _m ); this.reset(); }
		u.cameraPosition.value.setFromMatrixPosition( camera.matrixWorld );
		const fov = ( camera.fov ?? 50 ) * Math.PI / 180;
		const tanHalf = Math.tan( fov / 2 ), aspect = this._width / this._height;
		if ( tanHalf !== u.tanHalfFov.value || aspect !== u.aspect.value ) { u.tanHalfFov.value = tanHalf; u.aspect.value = aspect; this.reset(); }
		if ( options.tlas !== false || ! this._tlas ) this.updateTlas();
		u.jitter.value = path && this._jitter ? 1 : 0;
		u.reset.value = this._needsReset ? 1 : 0;
		if ( this._needsReset ) { this.sampleCount = 0; this._needsReset = false; }
		u.frame.value = ( this.frameCount ++ ) >>> 0;
		// one sample: the trace kernel then the shade kernel per bounce; a debug view traces the primary rays only
		const samples = path ? u.samplesPerFrame.value : 1, bounces = path ? u.maxBounces.value : 0;
		for ( let s = 0; s < samples; s ++ ) {

			u.sample.value = s;
			for ( let b = 0; b <= bounces; b ++ ) {

				u.bounce.value = b;
				renderer.compute( this.traceKernel );
				if ( path ) renderer.compute( this.shadeKernel );

			}

		}

		this.sampleCount += path ? samples : 0;

	}

	/** Draw the accumulated image to the current render target (a full-screen quad; pixels left of `split` are discarded). */
	render( renderer ) { this.quad.render( renderer ); }

	dispose() { this.traceKernel?.dispose(); this.shadeKernel?.dispose(); this.material.dispose(); this.buffers.frame?.dispose?.(); this.buffers.accel?.dispose?.(); this.buffers.tlas?.dispose?.(); }

	// --- kernels ------------------------------------------------------------------------------------------------------------

	_createKernels() {

		const mesh = this.mesh, u = this.uniforms, n = mesh.nodes, L = this.accel.layout, lit = this.lighting, materials = this.materials;
		const accel = storage( this.buffers.accel, 'vec4', this.accel.vec4Count );
		const tlas = storage( this.buffers.tlas, 'vec4', this.buffers.tlas.count );
		const frame = this.frameNode;
		const F = mesh.cornerFetch();
		const WH = this.pixelCount;
		const S_RAY = 3 * WH, S_DIR = 4 * WH, S_THROUGHPUT = 5 * WH, S_RADIANCE = 6 * WH, S_HIT = 7 * WH, S_HIT2 = 8 * WH;

		// --- helpers -----------------------------------------------------------------------------------------------------

		const safeInv = ( x ) => float( 1 ).div( select( abs( x ).lessThan( 1e-20 ), select( x.lessThan( 0 ), float( - 1e-20 ), float( 1e-20 ) ), x ) );
		const invDir = ( d ) => vec3( safeInv( d.x ), safeInv( d.y ), safeInv( d.z ) );
		const slab = ( mn, mx, o, inv, tMin, tMax ) => {

			const t0 = mn.sub( o ).mul( inv ), t1 = mx.sub( o ).mul( inv );
			const a = min( t0, t1 ), b = max( t0, t1 );
			const tn = max( max( a.x, a.y ), max( a.z, tMin ) ), tf = min( min( b.x, b.y ), min( b.z, tMax ) );
			return tn.lessThanEqual( tf );

		};
		const boxDistance = ( mn, mx, p ) => length( max( max( mn.sub( p ), p.sub( mx ) ), vec3( 0 ) ) );
		const boxFarDistance = ( mn, mx, p ) => length( max( abs( p.sub( mn ) ), abs( p.sub( mx ) ) ) );
		const xform = ( c0, c1, c2, c3, p ) => c0.xyz.mul( p.x ).add( c1.xyz.mul( p.y ) ).add( c2.xyz.mul( p.z ) ).add( c3.xyz );
		const xformDir = ( c0, c1, c2, d ) => c0.xyz.mul( d.x ).add( c1.xyz.mul( d.y ) ).add( c2.xyz.mul( d.z ) );
		// (M^-1)^T n from the inverse's columns
		const xformNormal = ( i0, i1, i2, nrm ) => normalize( vec3( dot( i0.xyz, nrm ), dot( i1.xyz, nrm ), dot( i2.xyz, nrm ) ) );
		const hashColor = ( seed ) => vec3( hash( seed.toUint() ).mul( 0.7 ).add( 0.25 ), hash( seed.add( 4096 ).toUint() ).mul( 0.7 ).add( 0.25 ), hash( seed.add( 8192 ).toUint() ).mul( 0.7 ).add( 0.25 ) );
		const heat = ( t ) => {

			const c0 = vec3( 0.12, 0.18, 0.55 ), c1 = vec3( 0.10, 0.70, 0.75 ), c2 = vec3( 0.95, 0.85, 0.25 ), c3 = vec3( 0.90, 0.20, 0.15 );
			const x = t.clamp( 0, 1 ).mul( 3 );
			return mix( mix( c0, c1, x.clamp( 0, 1 ) ), mix( c2, c3, x.sub( 2 ).clamp( 0, 1 ) ), x.sub( 1 ).clamp( 0, 1 ) );

		};

		const rand = pcgRand;

		/** Orthonormal basis around a unit normal (Duff et al. 2017). */
		const basis = ( nrm ) => {

			const s = select( nrm.z.greaterThanEqual( 0 ), float( 1 ), float( - 1 ) );
			const a = float( - 1 ).div( s.add( nrm.z ) ), b = nrm.x.mul( nrm.y ).mul( a );
			return {
				t: vec3( float( 1 ).add( s.mul( nrm.x ).mul( nrm.x ).mul( a ) ), s.mul( b ), s.negate().mul( nrm.x ) ),
				b: vec3( b, s.add( nrm.y.mul( nrm.y ).mul( a ) ), nrm.y.negate() ),
			};

		};

		const skyRadiance = ( d ) => mix( lit.groundColor, lit.skyColor, d.y.mul( 0.5 ).add( 0.5 ) );
		const PI = Math.PI;

		// --- the traversal (inlined once per kernel: closest hit in the trace kernel, any hit in the shade kernel) ----------
		// `hit`: { t, inst, cluster, tri, u, v, cost } vars; `anyHit` (JS boolean): stop at the first hit

		const trace = ( ro, rd, hit, anyHit ) => {

			const tMin = float( RAY_EPSILON );
			const rinv = invDir( rd ).toVar();
			const tstack = array( 'uint', STACK_TLAS ).toVar();
			const tsp = uint( 1 ).toVar();
			tstack.element( uint( 0 ) ).assign( uint( 0 ) );
			If( u.tlasInstanceCount.equal( uint( 0 ) ), () => { tsp.assign( uint( 0 ) ); } );

			Loop( { start: uint( 0 ), end: uint( MAX_STEPS ), type: 'uint', condition: '<', name: 'ti' }, () => {

				If( tsp.equal( uint( 0 ) ), () => { Break(); } );
				tsp.subAssign( uint( 1 ) );
				const node = tstack.element( tsp ).toVar();
				const n0 = tlas.element( node.mul( uint( TLAS_NODE_VEC4 ) ) ), n1 = tlas.element( node.mul( uint( TLAS_NODE_VEC4 ) ).add( uint( 1 ) ) );
				hit.cost.addAssign( float( 1 ) );
				If( slab( n0.xyz, n1.xyz, ro, rinv, tMin, hit.t ), () => {

					const word = floatBitsToUint( n0.w ).toVar();
					If( word.bitAnd( uint( LEAF_BIT ) ).equal( uint( 0 ) ), () => {

						tstack.element( tsp ).assign( word ); tsp.addAssign( uint( 1 ) );
						tstack.element( tsp ).assign( node.add( uint( 1 ) ) ); tsp.addAssign( uint( 1 ) );

					} ).Else( () => {

						const inst = word.bitAnd( uint( 0x7fffffff ) ).toVar();
						const rb = u.tlasInstanceBase.add( inst.mul( uint( TLAS_INSTANCE_VEC4 ) ) ).toVar();
						const m0 = tlas.element( rb ), m1 = tlas.element( rb.add( uint( 1 ) ) ), m2 = tlas.element( rb.add( uint( 2 ) ) ), m3 = tlas.element( rb.add( uint( 3 ) ) );
						const i0 = tlas.element( rb.add( uint( 4 ) ) ), i1 = tlas.element( rb.add( uint( 5 ) ) ), i2 = tlas.element( rb.add( uint( 6 ) ) ), i3 = tlas.element( rb.add( uint( 7 ) ) );
						const rec = tlas.element( rb.add( uint( 8 ) ) );
						const rec2 = tlas.element( rb.add( uint( 9 ) ) );
						// maxScale: the cut rule's (the largest column norm); pruneMax / pruneMin: bounds on the singular values, for the distance bounds of the prunes
						const root = floatBitsToUint( rec.x ), maxScale = rec.y.toVar(), pruneMax = rec2.x.toVar(), pruneMin = rec2.y.toVar();
						// the ray and the camera in object space (direction unnormalized: t is shared with world space)
						const po = xform( i0, i1, i2, i3, ro ).toVar(), pd = xformDir( i0, i1, i2, rd ).toVar();
						const pinv = invDir( pd ).toVar();
						const camObj = xform( i0, i1, i2, i3, u.cameraPosition ).toVar();
						const errScale = maxScale.mul( u.lodPixelScale ).toVar();
						const lodOn = u.enableLod.equal( uint( 1 ) ), forced = u.forceLevel.greaterThanEqual( 0 );

						const hstack = array( 'uint', STACK_HIERARCHY ).toVar();
						const hsp = uint( 1 ).toVar();
						hstack.element( uint( 0 ) ).assign( root );

						Loop( { start: uint( 0 ), end: uint( MAX_STEPS ), type: 'uint', condition: '<', name: 'hi' }, () => {

							If( hsp.equal( uint( 0 ) ), () => { Break(); } );
							hsp.subAssign( uint( 1 ) );
							const hn = hstack.element( hsp ).toVar();
							const hb = uint( L.hierarchyBase ).add( hn.mul( uint( HIER_VEC4 ) ) ).toVar();
							const h0 = accel.element( hb ), h1 = accel.element( hb.add( uint( 1 ) ) );
							hit.cost.addAssign( float( 1 ) );
							If( slab( h0.xyz, h1.xyz, po, pinv, tMin, hit.t ), () => {

								const hword = floatBitsToUint( h0.w ).toVar();
								If( hword.bitAnd( uint( LEAF_BIT ) ).equal( uint( 0 ) ), () => {

									const visit = uint( 1 ).toVar();
									If( lodOn.and( forced.not() ), () => {

										const h2 = accel.element( hb.add( uint( 2 ) ) ), h3 = accel.element( hb.add( uint( 3 ) ) );
										// too fine: the largest parent error, projected at the nearest possible distance, is at or below the threshold
										const nearD = boxDistance( h2.xyz, h3.xyz, camObj );
										const nearW = max( pruneMin.mul( nearD ).sub( h3.w.mul( maxScale ) ), u.lodNear );
										const parentUpper = h1.w.mul( errScale ).div( nearW );
										// too coarse: the smallest own error, projected at the farthest possible distance, is above it
										const farD = boxFarDistance( h2.xyz, h3.xyz, camObj );
										const farW = max( pruneMax.mul( farD ), u.lodNear );
										const ownLower = h2.w.mul( errScale ).div( farW );
										If( parentUpper.lessThanEqual( u.lodThreshold ).or( ownLower.greaterThan( u.lodThreshold ) ), () => { visit.assign( uint( 0 ) ); } );

									} );
									If( visit.equal( uint( 1 ) ), () => {

										hstack.element( hsp ).assign( hword ); hsp.addAssign( uint( 1 ) );
										hstack.element( hsp ).assign( hn.add( uint( 1 ) ) ); hsp.addAssign( uint( 1 ) );

									} );

								} ).Else( () => {

									const c = hword.bitAnd( uint( 0x7fffffff ) ).toVar();
									// the cut rule, exactly as the cull kernel evaluates it (world-space spheres, maxScale, the near clamp)
									const selected = uint( 1 ).toVar();
									If( lodOn, () => {

										const mb = c.mul( uint( META_STRIDE_VEC4 ) );
										const lodInfo = n.meta.element( mb.add( uint( META_LOD_INFO ) ) );
										If( forced, () => {

											selected.assign( select( abs( lodInfo.z.sub( u.forceLevel ) ).lessThan( 0.5 ), uint( 1 ), uint( 0 ) ) );

										} ).Else( () => {

											const ownSphere = n.meta.element( mb.add( uint( META_OWN_SPHERE ) ) ), parentSphere = n.meta.element( mb.add( uint( META_PARENT_SPHERE ) ) );
											const ownCenter = xform( m0, m1, m2, m3, ownSphere.xyz );
											const ownDist = max( length( ownCenter.sub( u.cameraPosition ) ).sub( ownSphere.w.mul( maxScale ) ), u.lodNear );
											const ownProjected = lodInfo.x.mul( maxScale ).mul( u.lodPixelScale ).div( ownDist );
											const parentCenter = xform( m0, m1, m2, m3, parentSphere.xyz );
											const parentDist = max( length( parentCenter.sub( u.cameraPosition ) ).sub( parentSphere.w.mul( maxScale ) ), u.lodNear );
											const parentProjected = lodInfo.y.mul( maxScale ).mul( u.lodPixelScale ).div( parentDist );
											selected.assign( select( ownProjected.lessThanEqual( u.lodThreshold ).and( parentProjected.greaterThan( u.lodThreshold ) ), uint( 1 ), uint( 0 ) ) );

										} );

									} );

									If( selected.equal( uint( 1 ) ), () => {

										const tb = uint( L.tableBase ).add( c );
										const table = accel.element( tb );
										const nodeBase = floatBitsToUint( table.x ), orderBase = floatBitsToUint( table.y );
										const cstack = array( 'uint', STACK_CLUSTER ).toVar();
										const csp = uint( 1 ).toVar();
										cstack.element( uint( 0 ) ).assign( nodeBase );

										Loop( { start: uint( 0 ), end: uint( 512 ), type: 'uint', condition: '<', name: 'ci' }, () => {

											If( csp.equal( uint( 0 ) ), () => { Break(); } );
											csp.subAssign( uint( 1 ) );
											const cn = cstack.element( csp ).toVar();
											const cb = uint( L.clusterBvhBase ).add( cn.mul( uint( CBVH_VEC4 ) ) );
											const b0 = accel.element( cb ), b1 = accel.element( cb.add( uint( 1 ) ) );
											hit.cost.addAssign( float( 1 ) );
											If( slab( b0.xyz, b1.xyz, po, pinv, tMin, hit.t ), () => {

												const cword = floatBitsToUint( b0.w ).toVar(), count = floatBitsToUint( b1.w ).toVar();
												If( count.equal( uint( 0 ) ), () => {

													cstack.element( csp ).assign( cword ); csp.addAssign( uint( 1 ) );
													cstack.element( csp ).assign( cn.add( uint( 1 ) ) ); csp.addAssign( uint( 1 ) );

												} ).Else( () => {

													Loop( { start: uint( 0 ), end: count, type: 'uint', condition: '<', name: 'k' }, ( { k } ) => {

														// order byte: 4 per u32, 16 per vec4
														const ob = orderBase.add( cword ).add( k );
														const ow = accel.element( uint( L.orderBase ).add( ob.shiftRight( uint( 4 ) ) ) );
														const comp = ob.shiftRight( uint( 2 ) ).bitAnd( uint( 3 ) );
														const wordV = select( comp.equal( uint( 0 ) ), ow.x, select( comp.equal( uint( 1 ) ), ow.y, select( comp.equal( uint( 2 ) ), ow.z, ow.w ) ) );
														const tri = floatBitsToUint( wordV ).shiftRight( ob.bitAnd( uint( 3 ) ).mul( uint( 8 ) ) ).bitAnd( uint( 255 ) ).toVar();
														hit.cost.addAssign( float( 1 ) );
														const v0 = F.fetch( c, tri, uint( 0 ) ).objectPosition.toVar();
														const v1 = F.fetch( c, tri, uint( 1 ) ).objectPosition.toVar();
														const v2 = F.fetch( c, tri, uint( 2 ) ).objectPosition.toVar();
														// Möller–Trumbore
														const e1 = v1.sub( v0 ), e2 = v2.sub( v0 );
														const pv = cross( pd, e2 );
														const det = dot( e1, pv ).toVar();
														const ok = select( u.cullBackFaces.equal( uint( 1 ) ), det.greaterThanEqual( 1e-12 ), abs( det ).greaterThanEqual( 1e-12 ) );
														If( ok, () => {

															const id = float( 1 ).div( det );
															const s = po.sub( v0 );
															const uu = dot( s, pv ).mul( id );
															const qv = cross( s, e1 );
															const vv = dot( pd, qv ).mul( id );
															const t = dot( e2, qv ).mul( id );
															If( uu.greaterThanEqual( 0 ).and( uu.lessThanEqual( 1 ) ).and( vv.greaterThanEqual( 0 ) ).and( uu.add( vv ).lessThanEqual( 1 ) ).and( t.greaterThan( tMin ) ).and( t.lessThan( hit.t ) ), () => {

																// any-hit: t = 0 fails every later box test, so the stacks drain at once
																if ( anyHit ) hit.t.assign( float( 0 ) );
																else { hit.t.assign( t ); hit.inst.assign( inst ); hit.cluster.assign( c ); hit.tri.assign( tri ); hit.u.assign( uu ); hit.v.assign( vv ); }

															} );

														} );

													} );

												} );

											} );

										} );

									} );

								} );

							} );

						} );

					} );

				} );

			} );

		};

		// --- the trace kernel: the path ray of this bounce (bounce 0 builds the camera ray and resets the path state) ----------

		const cameraRayNode = ( px, py, rng ) => {

			const jx = select( u.jitter.equal( uint( 1 ) ), rand( rng ), float( 0.5 ) ), jy = select( u.jitter.equal( uint( 1 ) ), rand( rng ), float( 0.5 ) );
			const nx = px.toFloat().add( jx ).div( u.width.toFloat() ).mul( 2 ).sub( 1 ).mul( u.tanHalfFov ).mul( u.aspect );
			const ny = float( 1 ).sub( py.toFloat().add( jy ).div( u.height.toFloat() ).mul( 2 ) ).mul( u.tanHalfFov );
			const cw = u.cameraWorld;
			return { origin: cw.element( 3 ).xyz, direction: normalize( xformDir( cw.element( 0 ), cw.element( 1 ), cw.element( 2 ), vec3( nx, ny, - 1 ) ) ) };

		};

		this.traceKernel = Fn( () => {

			const id = instanceIndex;
			If( id.lessThan( uint( WH ) ), () => {

				const ro = vec3( 0 ).toVar(), rd = vec3( 0 ).toVar(), rng = uint( 0 ).toVar(), done = uint( 0 ).toVar();
				If( u.bounce.equal( uint( 0 ) ), () => {

					const px = id.mod( u.width ), py = id.div( u.width );
					rng.assign( id.add( uint( 1 ) ).mul( uint( 0x9E3779B9 ) ).bitXor( u.frame.mul( uint( 0x85EBCA6B ) ) ).bitXor( u.sample.mul( uint( 0xC2B2AE35 ) ) ) );
					const ray = cameraRayNode( px, py, rng );
					ro.assign( ray.origin ); rd.assign( ray.direction );
					frame.element( id.add( uint( S_THROUGHPUT ) ) ).assign( vec4( 1, 1, 1, 0 ) );
					frame.element( id.add( uint( S_RADIANCE ) ) ).assign( vec4( 0 ) );
					If( u.reset.equal( uint( 1 ) ).and( u.sample.equal( uint( 0 ) ) ), () => { frame.element( id ).assign( vec4( 0 ) ); } );

				} ).Else( () => {

					const s0 = frame.element( id.add( uint( S_RAY ) ) ), s1 = frame.element( id.add( uint( S_DIR ) ) );
					ro.assign( s0.xyz ); done.assign( floatBitsToUint( s0.w ) ); rd.assign( s1.xyz ); rng.assign( floatBitsToUint( s1.w ) );

				} );

				If( done.equal( uint( 0 ) ), () => {

					const hit = { t: float( 1e30 ).toVar(), inst: uint( 0 ).toVar(), cluster: uint( 0 ).toVar(), tri: uint( 0 ).toVar(), u: float( 0 ).toVar(), v: float( 0 ).toVar(), cost: float( 0 ).toVar() };
					trace( ro, rd, hit, false );
					frame.element( id.add( uint( S_HIT ) ) ).assign( vec4( hit.t, uintBitsToFloat( hit.inst ), uintBitsToFloat( hit.cluster ), uintBitsToFloat( hit.tri ) ) );
					frame.element( id.add( uint( S_HIT2 ) ) ).assign( vec4( hit.u, hit.v, hit.cost, 0 ) );
					If( u.bounce.equal( uint( 0 ) ).and( u.sample.equal( uint( 0 ) ) ), () => {

						const missed = hit.t.greaterThanEqual( 1e30 );
						frame.element( id.add( uint( WH ) ) ).assign( select( missed, vec4( 0, 0, 0, hit.cost ), vec4( hit.inst.add( uint( 1 ) ).toFloat(), hit.cluster.toFloat(), hit.tri.toFloat(), hit.cost ) ) );
						frame.element( id.add( uint( 2 * WH ) ) ).assign( vec4( hit.u, hit.v, hit.t, 0 ) );

					} );

				} );

				frame.element( id.add( uint( S_RAY ) ) ).assign( vec4( ro, uintBitsToFloat( done ) ) );
				frame.element( id.add( uint( S_DIR ) ) ).assign( vec4( rd, uintBitsToFloat( rng ) ) );

			} );

		} )().compute( WH, [ this.workgroupSize ] );

		// --- the shade kernel: the hit's surface, the sun through a shadow ray, the next direction; a finished path is added
		// to the pixel's sum ---------------------------------------------------------------------------------------------------

		this.shadeKernel = Fn( () => {

			const id = instanceIndex;
			If( id.lessThan( uint( WH ) ), () => {

				const s0 = frame.element( id.add( uint( S_RAY ) ) ), s1 = frame.element( id.add( uint( S_DIR ) ) );
				const done = floatBitsToUint( s0.w ).toVar();
				If( done.equal( uint( 0 ) ), () => {

					const ro = s0.xyz.toVar(), rd = s1.xyz.toVar(), rng = floatBitsToUint( s1.w ).toVar();
					const throughput = frame.element( id.add( uint( S_THROUGHPUT ) ) ).xyz.toVar();
					const radiance = frame.element( id.add( uint( S_RADIANCE ) ) ).xyz.toVar();
					const h0 = frame.element( id.add( uint( S_HIT ) ) ), h1 = frame.element( id.add( uint( S_HIT2 ) ) );
					const hitT = h0.x.toVar(), hitInst = floatBitsToUint( h0.y ).toVar(), hitCluster = floatBitsToUint( h0.z ).toVar(), hitTri = floatBitsToUint( h0.w ).toVar();
					const hitU = h1.x, hitV = h1.y;

					If( hitT.greaterThanEqual( 1e30 ), () => {

						radiance.addAssign( throughput.mul( skyRadiance( rd ) ) );
						done.assign( uint( 1 ) );

					} ).Else( () => {

						// --- the surface: corners through the mesh's fetch, in the hit instance's object space -----------------------
						const rb = u.tlasInstanceBase.add( hitInst.mul( uint( TLAS_INSTANCE_VEC4 ) ) );
						const i0 = tlas.element( rb.add( uint( 4 ) ) ), i1 = tlas.element( rb.add( uint( 5 ) ) ), i2 = tlas.element( rb.add( uint( 6 ) ) );
						const f0 = F.fetch( hitCluster, hitTri, uint( 0 ) ), f1 = F.fetch( hitCluster, hitTri, uint( 1 ) ), f2 = F.fetch( hitCluster, hitTri, uint( 2 ) );
						const p0 = f0.objectPosition.toVar(), p1 = f1.objectPosition.toVar(), p2 = f2.objectPosition.toVar();
						const w0 = float( 1 ).sub( hitU ).sub( hitV );
						const P = ro.add( rd.mul( hitT ) ).toVar();
						const ng = xformNormal( i0, i1, i2, cross( p1.sub( p0 ), p2.sub( p0 ) ) ).toVar();
						const nsObj = f0.objectNormal.mul( w0 ).add( f1.objectNormal.mul( hitU ) ).add( f2.objectNormal.mul( hitV ) );
						const ns = xformNormal( i0, i1, i2, nsObj ).toVar();
						// two-sided: face the ray
						const flip = dot( ng, rd ).greaterThan( 0 );
						ng.assign( select( flip, ng.negate(), ng ) );
						ns.assign( select( flip, ns.negate(), ns ) );
						If( dot( ns, ng ).lessThan( 0 ), () => { ns.assign( ng ); } );
						const uvc = f0.objectUV ? f0.objectUV.mul( w0 ).add( f1.objectUV.mul( hitU ) ).add( f2.objectUV.mul( hitV ) ).toVar() : vec2( 0 ).toVar();
						const base = vec3( 0 ).toVar(), roughness = float( 0.5 ).toVar(), metalness = float( 0 ).toVar();
						if ( materials ) {

							const matId = materials.materialOf( hitInst );
							const g = vec2( 1e-9 );
							base.assign( materials.albedo( uvc, matId, [ g, g ], { level: true } ) );
							const surface = materials.surface( matId );
							roughness.assign( clamp( surface.roughness, 0.045, 1 ) ); metalness.assign( clamp( surface.metalness, 0, 1 ) );

						} else {

							base.assign( lit.flatColor ); roughness.assign( clamp( lit.roughness, 0.045, 1 ) ); metalness.assign( clamp( lit.metalness, 0, 1 ) );

						}

						const wo = rd.negate().toVar();
						const nv = max( dot( ns, wo ), 1e-4 ).toVar();
						const f0c = mix( vec3( 0.04 ), base, metalness ).toVar();
						const diffuseColor = base.mul( metalness.oneMinus() ).toVar();
						const a = roughness.mul( roughness ).toVar(), a2 = a.mul( a ).toVar();
						const brdf = ( wi ) => {

							// f = diffuse / π + F D V (the raster's lobe, radiometric)
							const h = normalize( wi.add( wo ) );
							const nl = max( dot( ns, wi ), 0 ), nh = max( dot( ns, h ), 0 ), vh = max( dot( wo, h ), 0 );
							const d = nh.mul( nh ).mul( a2.sub( 1 ) ).add( 1 );
							const D = a2.div( d.mul( d ).mul( PI ) );
							const gv = nl.mul( sqrt( nv.mul( nv ).mul( a2.oneMinus() ).add( a2 ) ) ), gl = nv.mul( sqrt( nl.mul( nl ).mul( a2.oneMinus() ).add( a2 ) ) );
							const V = float( 0.5 ).div( gv.add( gl ).max( 1e-5 ) );
							const Fr = f0c.add( f0c.oneMinus().mul( pow( max( vh.oneMinus(), 0 ), 5 ) ) );   // vh can round above 1: a negative pow base is indeterminate
							return diffuseColor.div( PI ).add( Fr.mul( D ).mul( V ) );

						};

						// --- the sun: a direction in the disk around the light, a shadow ray, irradiance π × lightColor (the raster's
						// convention: a Lambert surface facing the sun returns albedo × lightColor) ---------------------------------------
						const fr = basis( lit.lightDirection );
						const cosMax = cos( u.sunAngularRadius );
						const u1 = rand( rng ), u2 = rand( rng );
						const ct = float( 1 ).sub( u1.mul( float( 1 ).sub( cosMax ) ) ), st = sqrt( max( float( 1 ).sub( ct.mul( ct ) ), 0 ) ), ph = u2.mul( 2 * PI );
						const wiSun = normalize( fr.t.mul( st.mul( cos( ph ) ) ).add( fr.b.mul( st.mul( sin( ph ) ) ) ).add( lit.lightDirection.mul( ct ) ) ).toVar();
						const nlSun = dot( ns, wiSun ).toVar();
						If( nlSun.greaterThan( 0 ).and( dot( ng, wiSun ).greaterThan( 0 ) ), () => {

							const shadow = { t: float( 1e30 ).toVar(), cost: float( 0 ).toVar() };
							trace( P.add( ng.mul( RAY_EPSILON * 4 ) ), wiSun, shadow, true );
							If( shadow.t.greaterThanEqual( 1e30 ), () => {

								radiance.addAssign( throughput.mul( brdf( wiSun ) ).mul( lit.lightColor ).mul( PI ).mul( nlSun ) );

							} );

						} );

						// --- the next direction: a diffuse (cosine) or a specular (GGX visible normals) sample, weighted by the full
						// lobe pair (one-sample MIS over the lobes) ---------------------------------------------------------------------
						If( u.bounce.greaterThanEqual( u.maxBounces ), () => { done.assign( uint( 1 ) ); } ).Else( () => {

							const sb = basis( ns );
							const pS = clamp( mix( float( 0.08 ), float( 0.95 ), metalness ), 0.05, 0.95 ).toVar();
							const r0 = rand( rng ), r1 = rand( rng ), r2 = rand( rng );
							const wi = vec3( 0 ).toVar();
							const woL = vec3( dot( wo, sb.t ), dot( wo, sb.b ), dot( wo, ns ) ).toVar();
							If( r0.lessThan( pS ), () => {

								const vh = normalize( vec3( a.mul( woL.x ), a.mul( woL.y ), woL.z ) );
								const lensq = vh.x.mul( vh.x ).add( vh.y.mul( vh.y ) );
								const T1 = select( lensq.greaterThan( 0 ), vec3( vh.y.negate(), vh.x, 0 ).div( sqrt( max( lensq, 1e-20 ) ) ), vec3( 1, 0, 0 ) );
								const T2 = cross( vh, T1 );
								const rr = sqrt( r1 ), phi = r2.mul( 2 * PI );
								const t1 = rr.mul( cos( phi ) );
								const t2a = rr.mul( sin( phi ) );
								const ss = float( 0.5 ).mul( float( 1 ).add( vh.z ) );
								const t2 = ss.oneMinus().mul( sqrt( max( float( 1 ).sub( t1.mul( t1 ) ), 0 ) ) ).add( ss.mul( t2a ) );
								const nh = T1.mul( t1 ).add( T2.mul( t2 ) ).add( vh.mul( sqrt( max( float( 1 ).sub( t1.mul( t1 ) ).sub( t2.mul( t2 ) ), 0 ) ) ) );
								const hL = normalize( vec3( a.mul( nh.x ), a.mul( nh.y ), max( nh.z, 0 ) ) );
								const hW = sb.t.mul( hL.x ).add( sb.b.mul( hL.y ) ).add( ns.mul( hL.z ) );
								wi.assign( normalize( hW.mul( dot( wo, hW ).mul( 2 ) ).sub( wo ) ) );

							} ).Else( () => {

								const rr = sqrt( r1 ), phi = r2.mul( 2 * PI );
								wi.assign( normalize( sb.t.mul( rr.mul( cos( phi ) ) ).add( sb.b.mul( rr.mul( sin( phi ) ) ) ).add( ns.mul( sqrt( max( float( 1 ).sub( r1 ), 0 ) ) ) ) ) );

							} );

							const nl = dot( ns, wi ).toVar();
							If( nl.lessThanEqual( 0 ).or( dot( ng, wi ).lessThanEqual( 0 ) ), () => { done.assign( uint( 1 ) ); } ).Else( () => {

								const h = normalize( wi.add( wo ) );
								const nh = max( dot( ns, h ), 0 ), d = nh.mul( nh ).mul( a2.sub( 1 ) ).add( 1 );
								const D = a2.div( d.mul( d ).mul( PI ) );
								const G1 = nv.mul( 2 ).div( nv.add( sqrt( a2.add( a2.oneMinus().mul( nv ).mul( nv ) ) ) ) );
								const pdfS = G1.mul( D ).div( nv.mul( 4 ) ), pdfD = nl.div( PI );
								const pdf = pS.mul( pdfS ).add( pS.oneMinus().mul( pdfD ) ).max( 1e-6 );
								throughput.mulAssign( brdf( wi ).mul( nl ).div( pdf ) );
								// Russian roulette after the second bounce
								If( u.bounce.greaterThanEqual( uint( 2 ) ), () => {

									const q = clamp( max( max( throughput.x, throughput.y ), throughput.z ), 0.05, 0.95 );
									If( rand( rng ).greaterThan( q ), () => { throughput.assign( vec3( 0 ) ); } ).Else( () => { throughput.divAssign( q ); } );

								} );
								If( max( max( throughput.x, throughput.y ), throughput.z ).lessThanEqual( 0 ), () => { done.assign( uint( 1 ) ); } );
								ro.assign( P.add( ng.mul( RAY_EPSILON * 4 ) ) );
								rd.assign( wi );

							} );

						} );

					} );

					// a finished path joins the pixel's sum; a NaN or infinite sample (a rounding corner of a lobe) is dropped, and a huge one clamped
					// below the half-float range of the output, so one bad sample cannot black out a pixel until the next reset
					If( done.equal( uint( 1 ) ), () => {

						const finite = max( max( abs( radiance.x ), abs( radiance.y ) ), abs( radiance.z ) ).lessThan( 1e30 );
						frame.element( id ).addAssign( vec4( select( finite, min( radiance, vec3( 30000 ) ), vec3( 0 ) ), 1 ) );

					} );
					frame.element( id.add( uint( S_RAY ) ) ).assign( vec4( ro, uintBitsToFloat( done ) ) );
					frame.element( id.add( uint( S_DIR ) ) ).assign( vec4( rd, uintBitsToFloat( rng ) ) );
					frame.element( id.add( uint( S_THROUGHPUT ) ) ).assign( vec4( throughput, 0 ) );
					frame.element( id.add( uint( S_RADIANCE ) ) ).assign( vec4( radiance, 0 ) );

				} );

			} );

		} )().compute( WH, [ this.workgroupSize ] );

		// --- display: the accumulated image, or a debug view drawn from the primary hit record -----------------------------

		this.material.colorNode = Fn( () => {

			If( screenCoordinate.x.lessThan( u.split ), () => { Discard(); } );
			// the frame pixel under this viewport pixel (the frame may be smaller than the viewport)
			const px = min( floor( screenCoordinate.x.mul( u.width.toFloat() ).div( u.viewport.x ) ).toUint(), u.width.sub( uint( 1 ) ) ).toVar();
			const py = min( floor( screenCoordinate.y.mul( u.height.toFloat() ).div( u.viewport.y ) ).toUint(), u.height.sub( uint( 1 ) ) ).toVar();
			const p = py.mul( u.width ).add( px ).toVar();
			const color = vec3( 0 ).toVar();
			If( u.view.equal( uint( PathTraceView.PATH ) ), () => {

				const acc = frame.element( p );
				color.assign( min( acc.xyz.div( max( acc.w, 1 ) ).mul( u.exposure ), vec3( 60000 ) ) );

			} ).Else( () => {

				const rec = frame.element( p.add( uint( WH ) ) ), bary = frame.element( p.add( uint( 2 * WH ) ) );
				const inst1 = rec.x.toUint(), cluster = rec.y.toUint().toVar(), tri = rec.z.toUint().toVar(), cost = rec.w;
				If( u.view.equal( uint( PathTraceView.COST ) ), () => { color.assign( heat( cost.div( u.costMax ) ) ); } )
					.ElseIf( inst1.equal( uint( 0 ) ), () => { color.assign( u.background ); } )
					.Else( () => {

						const inst = inst1.sub( uint( 1 ) ).toVar();
						// the surface again: corners through the mesh's fetch, normals through the instance's inverse matrix
						const rb = u.tlasInstanceBase.add( inst.mul( uint( TLAS_INSTANCE_VEC4 ) ) );
						const i0 = tlas.element( rb.add( uint( 4 ) ) ), i1 = tlas.element( rb.add( uint( 5 ) ) ), i2 = tlas.element( rb.add( uint( 6 ) ) );
						const f0 = F.fetch( cluster, tri, uint( 0 ) ), f1 = F.fetch( cluster, tri, uint( 1 ) ), f2 = F.fetch( cluster, tri, uint( 2 ) );
						const w0 = float( 1 ).sub( bary.x ).sub( bary.y );
						const nsObj = f0.objectNormal.mul( w0 ).add( f1.objectNormal.mul( bary.x ) ).add( f2.objectNormal.mul( bary.y ) );
						const ns = xformNormal( i0, i1, i2, nsObj ).toVar();
						const ng = xformNormal( i0, i1, i2, cross( f1.objectPosition.sub( f0.objectPosition ), f2.objectPosition.sub( f0.objectPosition ) ) );
						// face the camera (the kernel flips the same way)
						const cw = u.cameraWorld;
						const nx = px.toFloat().add( 0.5 ).div( u.width.toFloat() ).mul( 2 ).sub( 1 ).mul( u.tanHalfFov ).mul( u.aspect );
						const ny = float( 1 ).sub( py.toFloat().add( 0.5 ).div( u.height.toFloat() ).mul( 2 ) ).mul( u.tanHalfFov );
						const rd = normalize( xformDir( cw.element( 0 ), cw.element( 1 ), cw.element( 2 ), vec3( nx, ny, - 1 ) ) );
						If( dot( ng, rd ).greaterThan( 0 ), () => { ns.assign( ns.negate() ); } );
						const lodInfo = n.meta.element( cluster.mul( uint( META_STRIDE_VEC4 ) ).add( uint( META_LOD_INFO ) ) );
						const col = vec3( 0 ).toVar();
						If( u.view.equal( uint( PathTraceView.NORMAL ) ), () => { col.assign( ns.mul( 0.5 ).add( 0.5 ) ); } )
							.ElseIf( u.view.equal( uint( PathTraceView.CLUSTER ) ), () => { col.assign( hashColor( cluster.toFloat() ) ); } )
							.ElseIf( u.view.equal( uint( PathTraceView.LEVEL ) ), () => { col.assign( mix( vec3( 0.16, 0.36, 0.85 ), vec3( 0.98, 0.80, 0.25 ), lodInfo.z.div( mesh.uniforms.maxLevel ) ) ); } )
							.ElseIf( u.view.equal( uint( PathTraceView.INSTANCE ) ), () => { col.assign( hashColor( inst.toFloat().add( 65536 ) ) ); } )
							.ElseIf( u.view.equal( uint( PathTraceView.TRIANGLE ) ), () => { col.assign( hashColor( cluster.mul( uint( mesh.meshletSet.options.maxTriangles ) ).add( tri ).toFloat().add( 1000000 ) ) ); } )
							.Else( () => {

								// ALBEDO: the material at the interpolated UV, with a mip level from the pixel's footprint on the surface: a ray cone,
								// the footprint width t × pixelAngle / cos, times the triangle's UV per world unit (explicit level: no derivatives in a branch)
								if ( materials ) {

									const uvc = f0.objectUV ? f0.objectUV.mul( w0 ).add( f1.objectUV.mul( bary.x ) ).add( f2.objectUV.mul( bary.y ) ) : vec2( 0 );
									const m0 = tlas.element( rb ), m1 = tlas.element( rb.add( uint( 1 ) ) ), m2 = tlas.element( rb.add( uint( 2 ) ) ), m3 = tlas.element( rb.add( uint( 3 ) ) );
									const q0 = xform( m0, m1, m2, m3, f0.objectPosition ), q1 = xform( m0, m1, m2, m3, f1.objectPosition ), q2 = xform( m0, m1, m2, m3, f2.objectPosition );
									const worldArea = length( cross( q1.sub( q0 ), q2.sub( q0 ) ) ).mul( 0.5 );
									const e1 = f1.objectUV.sub( f0.objectUV ), e2 = f2.objectUV.sub( f0.objectUV );
									const uvArea = abs( e1.x.mul( e2.y ).sub( e1.y.mul( e2.x ) ) ).mul( 0.5 );
									const pixelAngle = u.tanHalfFov.mul( 2 ).div( u.height.toFloat() );
									const footprint = bary.z.mul( pixelAngle ).div( max( abs( dot( ng, rd ) ), 0.05 ) );
									const f = footprint.mul( sqrt( uvArea.div( max( worldArea, 1e-12 ) ) ) );
									col.assign( materials.albedo( uvc, materials.materialOf( inst ), [ vec2( f, 0 ), vec2( 0, f ) ], { level: true } ) );

								} else col.assign( lit.flatColor );

							} );
						// the raster's own simple factor (the same function, so the halves of a split agree), none on the unlit albedo view
						const shade = select( u.view.equal( uint( PathTraceView.ALBEDO ) ), float( 1 ), simpleShade( lit, dot( ns, lit.lightDirection ) ) );
						color.assign( col.mul( shade ) );

					} );

			} );
			return vec4( color, 1 );

		} )();
		this.material.needsUpdate = true;

	}

}
