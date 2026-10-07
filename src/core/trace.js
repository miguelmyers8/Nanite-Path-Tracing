/**
 * CPU reference of the GPU ray traversal (`PathTracePass`): the same packed
 * buffers, the same stacks, the same LOD rule and the same intersection
 * arithmetic, in JavaScript, so a debugger can compare the two hit for hit.
 *
 * A ray is traced in world space through the TLAS, transformed into each
 * hit instance's object space (direction not normalized, so `t` means the
 * same in both spaces), and walked through the mesh's cluster hierarchy:
 *
 *   pop node; miss its geometry box → next
 *   internal: skip when every cluster below is too fine (max parent error
 *             projects at or below the threshold at the nearest possible
 *             distance) or too coarse (min own error projects above it at
 *             the farthest), else push both children
 *   leaf:     the cluster is in the cut iff own projected error <= threshold
 *             < parent projected error (Nanite's rule, `lodSelected`), then
 *             its triangle BVH is walked and triangles are tested with
 *             Möller–Trumbore
 *
 * @module trace
 */

import { lodSelected } from 'nanite/meshlets/core.js';
import { HIER_VEC4, CBVH_VEC4, TLAS_NODE_VEC4, TLAS_INSTANCE_VEC4, LEAF_BIT, cornerPosition } from './accel.js';

export const RAY_EPSILON = 1e-4;
export const STACK_TLAS = 24;
export const STACK_HIERARCHY = 48;
export const STACK_CLUSTER = 16;

/**
 * @typedef {Object} LodParams
 * @property {ArrayLike<number>} cameraPosition  world space; the cut is chosen from this point for every ray (primary and secondary alike)
 * @property {number} pixelScale                 screenHeight / (2 tan(fovY / 2))
 * @property {number} threshold                  pixels of projected error
 * @property {number} near                       distance clamp
 * @property {number} [forceLevel=-1]            draw exactly that level instead of the rule
 * @property {boolean} [lodTest=true]            false: every cluster is in the cut (sets without a DAG)
 */

/**
 * @typedef {Object} Hit
 * @property {number} t          along the world ray
 * @property {number} instance
 * @property {number} cluster    global cluster id
 * @property {number} triangle   local triangle index within the cluster
 * @property {number} u          barycentric weight of corner 1
 * @property {number} v          barycentric weight of corner 2
 */

/**
 * Trace one ray.
 *
 * @param {Object} ctx            { set, accel (buildAccel), tlas (buildInstanceTlas) }
 * @param {ArrayLike<number>} origin     world
 * @param {ArrayLike<number>} direction  world (unit)
 * @param {LodParams} lod
 * @param {Object} [options]
 * @param {number} [options.tMin=RAY_EPSILON]
 * @param {number} [options.tMax=Infinity]
 * @param {boolean} [options.anyHit=false]        stop at the first hit (shadow rays)
 * @param {boolean} [options.cullBackFaces=false]
 * @param {Object} [options.stats]                counters are added to it: nodes, clusters, selected, triangles
 * @returns {Hit|null}
 */
export function traceRay( ctx, origin, direction, lod, options = {} ) {

	const { set, accel, tlas } = ctx;
	const A = accel.data, AU = new Uint32Array( A.buffer, A.byteOffset, A.length ), L = accel.layout;
	const T = tlas.data, TU = new Uint32Array( T.buffer, T.byteOffset, T.length );
	const orderBytes = new Uint8Array( A.buffer, A.byteOffset + L.orderBase * 16 );
	const lodArray = accel.lod;
	const tMin = options.tMin ?? RAY_EPSILON;
	const anyHit = options.anyHit === true, cull = options.cullBackFaces === true;
	const stats = options.stats || null;
	const lodTest = lod.lodTest !== false;
	const forceLevel = lod.forceLevel ?? - 1;
	const cam = lod.cameraPosition;

	let closest = options.tMax ?? Infinity;
	let hit = null;

	const ox = origin[ 0 ], oy = origin[ 1 ], oz = origin[ 2 ], dx = direction[ 0 ], dy = direction[ 1 ], dz = direction[ 2 ];
	const ix = safeInv( dx ), iy = safeInv( dy ), iz = safeInv( dz );

	// --- TLAS ------------------------------------------------------------------------------------------------------------------
	const tstack = new Uint32Array( STACK_TLAS );
	let tsp = 0;
	tstack[ tsp ++ ] = 0;
	if ( tlas.instanceCount === 0 ) return null;

	while ( tsp > 0 ) {

		const node = tstack[ -- tsp ];
		const o = node * TLAS_NODE_VEC4 * 4;
		if ( stats ) stats.nodes ++;
		if ( ! slab( T[ o ], T[ o + 1 ], T[ o + 2 ], T[ o + 4 ], T[ o + 5 ], T[ o + 6 ], ox, oy, oz, ix, iy, iz, tMin, closest ) ) continue;
		const word = TU[ o + 3 ];
		if ( ( word & LEAF_BIT ) === 0 ) { tstack[ tsp ++ ] = word; tstack[ tsp ++ ] = node + 1; continue; }

		const inst = word & 0x7fffffff;
		const r = traceInstance( inst );
		if ( r && anyHit ) return r;

	}

	return hit;

	function traceInstance( inst ) {

		const ro = ( tlas.instanceBase + inst * TLAS_INSTANCE_VEC4 ) * 4;
		const M = T.subarray( ro, ro + 16 ), inv = T.subarray( ro + 16, ro + 32 );
		const root = TU[ ro + 32 ], maxScale = T[ ro + 33 ], minScale = T[ ro + 34 ];

		// the ray in object space (direction unnormalized so t is shared with world space); the camera too, for the LOD distances
		const pox = inv[ 0 ] * ox + inv[ 4 ] * oy + inv[ 8 ] * oz + inv[ 12 ];
		const poy = inv[ 1 ] * ox + inv[ 5 ] * oy + inv[ 9 ] * oz + inv[ 13 ];
		const poz = inv[ 2 ] * ox + inv[ 6 ] * oy + inv[ 10 ] * oz + inv[ 14 ];
		const pdx = inv[ 0 ] * dx + inv[ 4 ] * dy + inv[ 8 ] * dz;
		const pdy = inv[ 1 ] * dx + inv[ 5 ] * dy + inv[ 9 ] * dz;
		const pdz = inv[ 2 ] * dx + inv[ 6 ] * dy + inv[ 10 ] * dz;
		const pix = safeInv( pdx ), piy = safeInv( pdy ), piz = safeInv( pdz );
		const ccx = inv[ 0 ] * cam[ 0 ] + inv[ 4 ] * cam[ 1 ] + inv[ 8 ] * cam[ 2 ] + inv[ 12 ];
		const ccy = inv[ 1 ] * cam[ 0 ] + inv[ 5 ] * cam[ 1 ] + inv[ 9 ] * cam[ 2 ] + inv[ 13 ];
		const ccz = inv[ 2 ] * cam[ 0 ] + inv[ 6 ] * cam[ 1 ] + inv[ 10 ] * cam[ 2 ] + inv[ 14 ];
		const errScale = maxScale * lod.pixelScale;
		const scaleGap = ( maxScale - minScale );

		const hstack = new Uint32Array( STACK_HIERARCHY );
		let hsp = 0;
		hstack[ hsp ++ ] = root;
		let found = null;

		while ( hsp > 0 ) {

			const node = hstack[ -- hsp ];
			const o = ( L.hierarchyBase + node * HIER_VEC4 ) * 4;
			if ( stats ) stats.nodes ++;
			if ( ! slab( A[ o ], A[ o + 1 ], A[ o + 2 ], A[ o + 4 ], A[ o + 5 ], A[ o + 6 ], pox, poy, poz, pix, piy, piz, tMin, closest ) ) continue;
			const word = AU[ o + 3 ];

			if ( ( word & LEAF_BIT ) === 0 ) {

				if ( lodTest && forceLevel < 0 ) {

					// too fine: the largest parent error, projected at the nearest possible distance, is at or below the threshold
					const maxParentError = A[ o + 7 ], minOwnError = A[ o + 11 ], maxParentRadius = A[ o + 15 ];
					const nearD = boxDistance( A[ o + 8 ], A[ o + 9 ], A[ o + 10 ], A[ o + 12 ], A[ o + 13 ], A[ o + 14 ], ccx, ccy, ccz );
					const nearW = Math.max( minScale * nearD - maxParentRadius * maxScale - scaleGap * 0, lod.near );
					// (a non-uniform scale: minScale shrinks the distance, maxScale grows the radius; both already on the safe side)
					const parentUpper = maxParentError * errScale / nearW;
					if ( parentUpper <= lod.threshold ) continue;
					// too coarse: the smallest own error, projected at the farthest possible distance, is above the threshold
					const farD = boxFarDistance( A[ o + 8 ], A[ o + 9 ], A[ o + 10 ], A[ o + 12 ], A[ o + 13 ], A[ o + 14 ], ccx, ccy, ccz );
					const farW = Math.max( maxScale * farD, lod.near );
					const ownLower = minOwnError * errScale / farW;
					if ( ownLower > lod.threshold ) continue;

				}

				hstack[ hsp ++ ] = word; hstack[ hsp ++ ] = node + 1;
				continue;

			}

			const c = word & 0x7fffffff;
			if ( stats ) stats.clusters ++;
			if ( lodTest ) {

				const lo = c * 12;
				if ( forceLevel >= 0 ) { if ( lodArray[ lo + 10 ] !== forceLevel ) continue; }
				else if ( ! lodSelected( lodArray, lo, M, maxScale, cam[ 0 ], cam[ 1 ], cam[ 2 ], lod.pixelScale, lod.threshold, lod.near ) ) continue;

			}

			if ( stats ) stats.selected ++;
			const r = traceCluster( inst, c, pox, poy, poz, pdx, pdy, pdz, pix, piy, piz );
			if ( r ) { found = r; if ( anyHit ) return r; }

		}

		return found;

	}

	function traceCluster( inst, c, pox, poy, poz, pdx, pdy, pdz, pix, piy, piz ) {

		const t = ( L.tableBase + c ) * 4;
		const nodeBase = AU[ t ], orderBase = AU[ t + 1 ];
		const cstack = new Uint32Array( STACK_CLUSTER );
		let csp = 0;
		cstack[ csp ++ ] = nodeBase;
		let found = null;

		while ( csp > 0 ) {

			const node = cstack[ -- csp ];
			const o = ( L.clusterBvhBase + node * CBVH_VEC4 ) * 4;
			if ( stats ) stats.nodes ++;
			if ( ! slab( A[ o ], A[ o + 1 ], A[ o + 2 ], A[ o + 4 ], A[ o + 5 ], A[ o + 6 ], pox, poy, poz, pix, piy, piz, tMin, closest ) ) continue;
			const word = AU[ o + 3 ], count = AU[ o + 7 ];
			if ( count === 0 ) { cstack[ csp ++ ] = word; cstack[ csp ++ ] = node + 1; continue; }

			for ( let k = 0; k < count; k ++ ) {

				const tri = orderBytes[ orderBase + word + k ];
				if ( stats ) stats.triangles ++;
				cornerPosition( set, c, tri, 0, _v, 0 ); cornerPosition( set, c, tri, 1, _v, 3 ); cornerPosition( set, c, tri, 2, _v, 6 );
				const r = intersectTriangle( _v, pox, poy, poz, pdx, pdy, pdz, tMin, closest, cull );
				if ( r ) {

					closest = r.t;
					found = hit = { t: r.t, instance: inst, cluster: c, triangle: tri, u: r.u, v: r.v };
					if ( anyHit ) return found;

				}

			}

		}

		return found;

	}

}

const _v = new Float32Array( 9 );

/** 1 / d with zero guarded exactly like the kernel. */
export function safeInv( d ) {

	const a = Math.abs( d ) < 1e-20 ? ( d < 0 ? - 1e-20 : 1e-20 ) : d;
	return 1 / a;

}

/** Ray / box slab test. */
export function slab( x0, y0, z0, x1, y1, z1, ox, oy, oz, ix, iy, iz, tMin, tMax ) {

	const tx0 = ( x0 - ox ) * ix, tx1 = ( x1 - ox ) * ix;
	const ty0 = ( y0 - oy ) * iy, ty1 = ( y1 - oy ) * iy;
	const tz0 = ( z0 - oz ) * iz, tz1 = ( z1 - oz ) * iz;
	const tn = Math.max( Math.min( tx0, tx1 ), Math.min( ty0, ty1 ), Math.min( tz0, tz1 ), tMin );
	const tf = Math.min( Math.max( tx0, tx1 ), Math.max( ty0, ty1 ), Math.max( tz0, tz1 ), tMax );
	return tn <= tf;

}

/** Distance from a point to a box (0 inside). */
export function boxDistance( x0, y0, z0, x1, y1, z1, px, py, pz ) {

	const dx = Math.max( x0 - px, 0, px - x1 ), dy = Math.max( y0 - py, 0, py - y1 ), dz = Math.max( z0 - pz, 0, pz - z1 );
	return Math.sqrt( dx * dx + dy * dy + dz * dz );

}

/** Distance from a point to the farthest corner of a box. */
export function boxFarDistance( x0, y0, z0, x1, y1, z1, px, py, pz ) {

	const dx = Math.max( Math.abs( px - x0 ), Math.abs( px - x1 ) ), dy = Math.max( Math.abs( py - y0 ), Math.abs( py - y1 ) ), dz = Math.max( Math.abs( pz - z0 ), Math.abs( pz - z1 ) );
	return Math.sqrt( dx * dx + dy * dy + dz * dz );

}

/**
 * Möller–Trumbore. `v` holds the three corners (9 floats). Returns { t, u, v } or null.
 */
export function intersectTriangle( v, ox, oy, oz, dx, dy, dz, tMin, tMax, cull = false ) {

	const e1x = v[ 3 ] - v[ 0 ], e1y = v[ 4 ] - v[ 1 ], e1z = v[ 5 ] - v[ 2 ];
	const e2x = v[ 6 ] - v[ 0 ], e2y = v[ 7 ] - v[ 1 ], e2z = v[ 8 ] - v[ 2 ];
	const px = dy * e2z - dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y - dy * e2x;
	const det = e1x * px + e1y * py + e1z * pz;
	if ( cull ? det < 1e-12 : Math.abs( det ) < 1e-12 ) return null;
	const inv = 1 / det;
	const sx = ox - v[ 0 ], sy = oy - v[ 1 ], sz = oz - v[ 2 ];
	const u = ( sx * px + sy * py + sz * pz ) * inv;
	if ( u < 0 || u > 1 ) return null;
	const qx = sy * e1z - sz * e1y, qy = sz * e1x - sx * e1z, qz = sx * e1y - sy * e1x;
	const w = ( dx * qx + dy * qy + dz * qz ) * inv;
	if ( w < 0 || u + w > 1 ) return null;
	const t = ( e2x * qx + e2y * qy + e2z * qz ) * inv;
	if ( t <= tMin || t >= tMax ) return null;
	return { t, u, v: w };

}

/**
 * Brute force twin of `traceRay`: every cluster of every instance that
 * `lodSelected` puts in the cut, every triangle. For tests.
 */
export function traceRayBruteForce( ctx, origin, direction, lod, options = {} ) {

	const { set, tlas } = ctx;
	const T = tlas.data, TU = new Uint32Array( T.buffer, T.byteOffset, T.length );
	const lodArray = ctx.accel.lod;
	const tMin = options.tMin ?? RAY_EPSILON, cull = options.cullBackFaces === true;
	const lodTest = lod.lodTest !== false, forceLevel = lod.forceLevel ?? - 1, cam = lod.cameraPosition;
	const meshes = set.meshes || [ { first: 0, count: set.meshletCount } ];
	let closest = options.tMax ?? Infinity, hit = null;
	const ox = origin[ 0 ], oy = origin[ 1 ], oz = origin[ 2 ], dx = direction[ 0 ], dy = direction[ 1 ], dz = direction[ 2 ];

	for ( let inst = 0; inst < tlas.instanceCount; inst ++ ) {

		const ro = ( tlas.instanceBase + inst * TLAS_INSTANCE_VEC4 ) * 4;
		const M = T.subarray( ro, ro + 16 ), inv = T.subarray( ro + 16, ro + 32 ), maxScale = T[ ro + 33 ], k = TU[ ro + 35 ];
		const pox = inv[ 0 ] * ox + inv[ 4 ] * oy + inv[ 8 ] * oz + inv[ 12 ], poy = inv[ 1 ] * ox + inv[ 5 ] * oy + inv[ 9 ] * oz + inv[ 13 ], poz = inv[ 2 ] * ox + inv[ 6 ] * oy + inv[ 10 ] * oz + inv[ 14 ];
		const pdx = inv[ 0 ] * dx + inv[ 4 ] * dy + inv[ 8 ] * dz, pdy = inv[ 1 ] * dx + inv[ 5 ] * dy + inv[ 9 ] * dz, pdz = inv[ 2 ] * dx + inv[ 6 ] * dy + inv[ 10 ] * dz;
		const { first, count } = meshes[ k ];
		for ( let c = first; c < first + count; c ++ ) {

			if ( lodTest ) {

				const lo = c * 12;
				if ( forceLevel >= 0 ) { if ( lodArray[ lo + 10 ] !== forceLevel ) continue; }
				else if ( ! lodSelected( lodArray, lo, M, maxScale, cam[ 0 ], cam[ 1 ], cam[ 2 ], lod.pixelScale, lod.threshold, lod.near ) ) continue;

			}

			const tc = set.meshlets[ c * 4 + 3 ];
			for ( let tri = 0; tri < tc; tri ++ ) {

				cornerPosition( set, c, tri, 0, _v, 0 ); cornerPosition( set, c, tri, 1, _v, 3 ); cornerPosition( set, c, tri, 2, _v, 6 );
				const r = intersectTriangle( _v, pox, poy, poz, pdx, pdy, pdz, tMin, closest, cull );
				if ( r ) { closest = r.t; hit = { t: r.t, instance: inst, cluster: c, triangle: tri, u: r.u, v: r.v }; }

			}

		}

	}

	return hit;

}

/**
 * The primary ray of pixel (px, py) exactly as the kernel builds it: through the pixel centre (plus a jitter in [0, 1)²)
 * of a `width × height` image, from a camera's world matrix (column-major, 16) with a vertical field of view.
 * @returns {{ origin: number[], direction: number[] }}
 */
export function cameraRay( cameraWorld, fovYRadians, width, height, px, py, jx = 0.5, jy = 0.5 ) {

	const tanHalf = Math.tan( fovYRadians / 2 ), aspect = width / height;
	const nx = ( ( px + jx ) / width * 2 - 1 ) * tanHalf * aspect;
	const ny = ( 1 - ( py + jy ) / height * 2 ) * tanHalf;
	const m = cameraWorld;
	let x = m[ 0 ] * nx + m[ 4 ] * ny - m[ 8 ], y = m[ 1 ] * nx + m[ 5 ] * ny - m[ 9 ], z = m[ 2 ] * nx + m[ 6 ] * ny - m[ 10 ];
	const l = Math.hypot( x, y, z ); x /= l; y /= l; z /= l;
	return { origin: [ m[ 12 ], m[ 13 ], m[ 14 ] ], direction: [ x, y, z ] };

}
