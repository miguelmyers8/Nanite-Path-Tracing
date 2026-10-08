/**
 * Acceleration structures for path tracing a Nanite-style cluster set.
 *
 * Three levels, all traversed with explicit stacks:
 *
 *  1. Cluster BVHs (static, offline): one small binary BVH over the triangles
 *     of every cluster of every LOD level, in object space. Leaves hold up to
 *     `leafTriangles` triangles through a per-cluster order list. This is the
 *     cluster-level acceleration structure of RTX Mega Geometry (a CLAS)
 *     built in software.
 *
 *  2. The cluster hierarchy (static, offline, one per mesh): a BVH over ALL
 *     clusters of all LOD levels (one sub-tree per level, joined at the top),
 *     with the LOD data a ray needs to pick the cut while it traverses:
 *
 *       maxParentError   the largest parent error in the subtree: when it
 *                        projects at or below the threshold nothing in the
 *                        subtree is in the cut (everything is too fine), skip
 *       minOwnError      the smallest own error in the subtree: when it
 *                        projects above the threshold everything is too coarse, skip
 *       lodBox           the box of the subtree's own and parent sphere centres,
 *                        for conservative near / far distances in those tests
 *       maxParentRadius  for the distance term under a non-uniform scale
 *
 *     A leaf is one cluster; the ray applies the exact cut rule of the cull
 *     kernel there ("own projected error <= threshold < parent projected
 *     error", with the same spheres, scale and distance clamp), so the traced
 *     surface is the rasterized surface. No per-frame build: the hierarchy is
 *     a property of the set, like Nanite's own culling hierarchy.
 *
 *  3. The TLAS (per frame, CPU): a BVH over the instances' world boxes, with
 *     a record per instance (world and inverse matrices, scales, the mesh's
 *     hierarchy root). Instances are few and the CPU already owns their
 *     matrices.
 *
 * Everything is packed into vec4 arrays exactly as the GPU reads them; the
 * CPU reference (`trace.js`) reads the same packed arrays.
 *
 * Packed accel buffer (vec4 slots, u32 fields as bit patterns):
 *
 *   hierarchy node (4 vec4): [geomMin.xyz, word] [geomMax.xyz, maxParentError] [lodMin.xyz, minOwnError] [lodMax.xyz, maxParentRadius]
 *     word: bit 31 set = leaf, low bits = cluster id; else = right child node index (left = node + 1)
 *   cluster BVH node (2 vec4): [min.xyz, word] [max.xyz, count]
 *     count 0: internal, word = right child (absolute node index); else leaf, word = first order entry (local to the cluster)
 *   order words: 4 bytes per vec4 component, byte k of cluster c at firstTriangle[c] + k = local triangle index
 *   cluster table (1 vec4): [nodeBase, orderBase (= firstTriangle), nodeCount, level]
 *
 * @module accel
 */

import { buildBvh } from './bvh.js';

export const HIER_VEC4 = 4;
export const CBVH_VEC4 = 2;
export const TABLE_VEC4 = 1;
export const TLAS_NODE_VEC4 = 2;
export const TLAS_INSTANCE_VEC4 = 10;
export const LEAF_BIT = 0x80000000;
/** the TLAS traversal stack of the kernel and the CPU twin (trace.js STACK_TLAS) */
const TLAS_STACK_DEPTH = 24;

export const DEFAULT_ACCEL_OPTIONS = Object.freeze( {
	/** triangles per cluster BVH leaf */
	leafTriangles: 4,
	/** clusters per hierarchy leaf (1: the leaf is the cluster) */
	leafClusters: 1,
} );

/**
 * LOD array of a set (12 floats per cluster, LodDagBuilder layout), or the
 * "no LOD" array (own error 0, parent error +inf) when the set has none.
 * Mirrors MeshletMesh.lodArray without the three.js dependency.
 */
export function lodArrayOf( set ) {

	if ( set.lod && set.lod.length === set.meshletCount * 12 ) return set.lod;
	const lod = new Float32Array( Math.max( 1, set.meshletCount ) * 12 );
	for ( let i = 0; i < set.meshletCount; i ++ ) {

		const o = i * 12, b = i * 12;
		for ( let k = 0; k < 4; k ++ ) { lod[ o + k ] = set.bounds[ b + k ]; lod[ o + 4 + k ] = set.bounds[ b + k ]; }
		lod[ o + 9 ] = Infinity;

	}

	return lod;

}

/**
 * Object-space position of corner `corner` of local triangle `tri` of cluster `c` (the CPU twin of the mesh's corner fetch
 * for uncompressed sets). Writes xyz into `out` at `oo`.
 */
export function cornerPosition( set, c, tri, corner, out, oo = 0 ) {

	const vo = set.meshlets[ c * 4 ], to = set.meshlets[ c * 4 + 1 ];
	const local = set.triangles[ to + tri * 3 + corner ];
	const v = set.vertices[ vo + local ] * 3;
	out[ oo ] = set.positions[ v ]; out[ oo + 1 ] = set.positions[ v + 1 ]; out[ oo + 2 ] = set.positions[ v + 2 ];
	return out;

}

const _p = new Float32Array( 9 );

/**
 * Build the cluster BVHs and the per-mesh hierarchies of a set and pack them.
 *
 * @param {Object} set  an uncompressed MeshletSet (with or without `lod`), or a merged scene set (`set.meshes`)
 * @param {Partial<typeof DEFAULT_ACCEL_OPTIONS>} [options]
 * @returns {AccelResult}
 */
export function buildAccel( set, options = {} ) {

	if ( set.pooled || set.chunkOrigins ) throw new Error( 'buildAccel: cluster pool (chunked) sets are not supported yet' );
	const opts = { ...DEFAULT_ACCEL_OPTIONS, ...options };
	const t0 = now();
	const n = set.meshletCount;
	const lod = lodArrayOf( set );
	const meshes = set.meshes || [ { first: 0, count: n } ];
	// the order table stores a cluster's local triangle index in one byte
	for ( let c = 0; c < n; c ++ ) if ( set.meshlets[ c * 4 + 3 ] > 256 ) throw new Error( `buildAccel: cluster ${ c } has ${ set.meshlets[ c * 4 + 3 ] } triangles; the tracer supports at most 256 triangles per cluster (build the set with maxTriangles <= 256)` );

	// --- 1. cluster BVHs -------------------------------------------------------------------------------------------------

	const clusterBox = new Float32Array( n * 6 );
	const triangleTotal = set.firstTriangle ? set.firstTriangle[ n ] : set.triangleCount;
	const order = new Uint8Array( ( triangleTotal + 3 ) & ~ 3 );
	const bvhs = new Array( n );
	let cbvhNodes = 0, maxClusterDepth = 0;
	let triBoxes = new Float32Array( 0 ), triCentroids = new Float32Array( 0 );

	for ( let c = 0; c < n; c ++ ) {

		const tc = set.meshlets[ c * 4 + 3 ];
		if ( triBoxes.length < tc * 6 ) { triBoxes = new Float32Array( tc * 6 ); triCentroids = new Float32Array( tc * 3 ); }
		for ( let t = 0; t < tc; t ++ ) {

			cornerPosition( set, c, t, 0, _p, 0 ); cornerPosition( set, c, t, 1, _p, 3 ); cornerPosition( set, c, t, 2, _p, 6 );
			const b = t * 6;
			for ( let a = 0; a < 3; a ++ ) {

				const lo = Math.min( _p[ a ], _p[ 3 + a ], _p[ 6 + a ] ), hi = Math.max( _p[ a ], _p[ 3 + a ], _p[ 6 + a ] );
				triBoxes[ b + a ] = lo; triBoxes[ b + 3 + a ] = hi;
				triCentroids[ t * 3 + a ] = ( lo + hi ) * 0.5;

			}

		}

		const bvh = buildBvh( tc, triBoxes, triCentroids, { leafSize: opts.leafTriangles, maxDepth: 14 } );
		bvhs[ c ] = bvh;
		cbvhNodes += bvh.nodeCount;
		if ( bvh.depth > maxClusterDepth ) maxClusterDepth = bvh.depth;
		const ft = set.firstTriangle ? set.firstTriangle[ c ] : triangleOffsetOf( set, c );
		for ( let t = 0; t < tc; t ++ ) order[ ft + t ] = bvh.order[ t ];
		const cb = c * 6;
		if ( tc > 0 ) { clusterBox.set( bvh.min.subarray( 0, 3 ), cb ); clusterBox.set( bvh.max.subarray( 0, 3 ), cb + 3 ); }
		else { const s = c * 12; clusterBox[ cb ] = clusterBox[ cb + 1 ] = clusterBox[ cb + 2 ] = set.bounds[ s ]; clusterBox[ cb + 3 ] = clusterBox[ cb + 4 ] = clusterBox[ cb + 5 ] = set.bounds[ s ]; }

	}

	// --- 2. hierarchies, one per mesh -------------------------------------------------------------------------------------

	const hier = [];        // node records { min[3], max[3], word, maxParentError, lodMin[3], lodMax[3], minOwnError, maxParentRadius }
	const meshRoot = new Uint32Array( meshes.length );
	let maxHierDepth = 0;
	for ( let k = 0; k < meshes.length; k ++ ) {

		const { first, count } = meshes[ k ];
		meshRoot[ k ] = hier.length;
		const r = buildMeshHierarchy( set, lod, clusterBox, first, count, opts, hier );
		if ( r.depth > maxHierDepth ) maxHierDepth = r.depth;

	}

	// --- 3. pack ------------------------------------------------------------------------------------------------------------

	const hierarchyBase = 0;
	const clusterBvhBase = hierarchyBase + hier.length * HIER_VEC4;
	const orderBase = clusterBvhBase + cbvhNodes * CBVH_VEC4;
	const orderVec4 = Math.ceil( order.length / 16 );
	const tableBase = orderBase + orderVec4;
	const vec4Count = tableBase + Math.max( 1, n ) * TABLE_VEC4;
	const data = new Float32Array( vec4Count * 4 );
	const u = new Uint32Array( data.buffer );

	for ( let i = 0; i < hier.length; i ++ ) {

		const h = hier[ i ], o = ( hierarchyBase + i * HIER_VEC4 ) * 4;
		data[ o ] = h.min[ 0 ]; data[ o + 1 ] = h.min[ 1 ]; data[ o + 2 ] = h.min[ 2 ]; u[ o + 3 ] = h.word >>> 0;
		data[ o + 4 ] = h.max[ 0 ]; data[ o + 5 ] = h.max[ 1 ]; data[ o + 6 ] = h.max[ 2 ]; data[ o + 7 ] = h.maxParentError;
		data[ o + 8 ] = h.lodMin[ 0 ]; data[ o + 9 ] = h.lodMin[ 1 ]; data[ o + 10 ] = h.lodMin[ 2 ]; data[ o + 11 ] = h.minOwnError;
		data[ o + 12 ] = h.lodMax[ 0 ]; data[ o + 13 ] = h.lodMax[ 1 ]; data[ o + 14 ] = h.lodMax[ 2 ]; data[ o + 15 ] = h.maxParentRadius;

	}

	const clusterNodeBase = new Uint32Array( n );
	let nb = 0;
	for ( let c = 0; c < n; c ++ ) {

		const bvh = bvhs[ c ];
		clusterNodeBase[ c ] = nb;
		for ( let i = 0; i < bvh.nodeCount; i ++ ) {

			const o = ( clusterBvhBase + ( nb + i ) * CBVH_VEC4 ) * 4;
			data[ o ] = bvh.min[ i * 3 ]; data[ o + 1 ] = bvh.min[ i * 3 + 1 ]; data[ o + 2 ] = bvh.min[ i * 3 + 2 ];
			data[ o + 4 ] = bvh.max[ i * 3 ]; data[ o + 5 ] = bvh.max[ i * 3 + 1 ]; data[ o + 6 ] = bvh.max[ i * 3 + 2 ];
			if ( bvh.right[ i ] < 0 ) { u[ o + 3 ] = bvh.first[ i ]; u[ o + 7 ] = bvh.count[ i ]; }
			else { u[ o + 3 ] = nb + bvh.right[ i ]; u[ o + 7 ] = 0; }

		}

		const t = ( tableBase + c ) * 4;
		u[ t ] = nb; u[ t + 1 ] = set.firstTriangle ? set.firstTriangle[ c ] : triangleOffsetOf( set, c ); u[ t + 2 ] = bvh.nodeCount; u[ t + 3 ] = lod[ c * 12 + 10 ];
		nb += bvh.nodeCount;

	}

	new Uint8Array( data.buffer, orderBase * 16, order.length ).set( order );

	return {
		data, vec4Count,
		layout: { hierarchyBase, clusterBvhBase, orderBase, tableBase, hierarchyNodeCount: hier.length, clusterBvhNodeCount: cbvhNodes },
		meshRoot, clusterNodeBase, clusterBox, order,
		maxHierarchyDepth: maxHierDepth, maxClusterDepth, clusterCount: n, lod,
		buildTimeMs: now() - t0,
	};

}

/**
 * @typedef {Object} AccelResult
 * @property {Float32Array} data        the packed buffer (vec4 slots; u32 fields as bit patterns)
 * @property {number} vec4Count
 * @property {Object} layout            { hierarchyBase, clusterBvhBase, orderBase, tableBase, hierarchyNodeCount, clusterBvhNodeCount } in vec4 units
 * @property {Uint32Array} meshRoot     hierarchy root node per mesh
 * @property {Uint32Array} clusterNodeBase
 * @property {Float32Array} clusterBox  6 floats per cluster, object space
 * @property {number} maxHierarchyDepth
 * @property {number} maxClusterDepth
 */

function triangleOffsetOf( set, c ) {

	let t = 0;
	for ( let i = 0; i < c; i ++ ) t += set.meshlets[ i * 4 + 3 ];
	return t;

}

/** The hierarchy of one mesh (clusters [first, first + count)): a BVH per level, joined by a BVH over the level roots. */
function buildMeshHierarchy( set, lod, clusterBox, first, count, opts, out ) {

	// clusters per level; a cluster without triangles has nothing to hit and is left out
	const byLevel = new Map();
	for ( let c = first; c < first + count; c ++ ) {

		if ( set.meshlets[ c * 4 + 3 ] === 0 ) continue;
		const L = lod[ c * 12 + 10 ] | 0;
		if ( ! byLevel.has( L ) ) byLevel.set( L, [] );
		byLevel.get( L ).push( c );

	}

	if ( byLevel.size === 0 ) throw new Error( `buildAccel: the mesh with clusters [${ first }, ${ first + count }) has no non-empty clusters` );
	const levels = Array.from( byLevel.keys() ).sort( ( a, b ) => a - b );
	const subtrees = [];
	for ( const L of levels ) {

		const ids = byLevel.get( L );
		const boxes = new Float32Array( ids.length * 6 ), centroids = new Float32Array( ids.length * 3 );
		for ( let i = 0; i < ids.length; i ++ ) {

			const c = ids[ i ];
			boxes.set( clusterBox.subarray( c * 6, c * 6 + 6 ), i * 6 );
			for ( let a = 0; a < 3; a ++ ) centroids[ i * 3 + a ] = ( clusterBox[ c * 6 + a ] + clusterBox[ c * 6 + 3 + a ] ) * 0.5;

		}

		subtrees.push( { bvh: buildBvh( ids.length, boxes, centroids, { leafSize: opts.leafClusters, maxDepth: 36 } ), ids } );

	}

	// top: a BVH over the level roots
	const tb = new Float32Array( subtrees.length * 6 ), tcen = new Float32Array( subtrees.length * 3 );
	for ( let i = 0; i < subtrees.length; i ++ ) {

		const b = subtrees[ i ].bvh;
		tb.set( b.min.subarray( 0, 3 ), i * 6 ); tb.set( b.max.subarray( 0, 3 ), i * 6 + 3 );
		for ( let a = 0; a < 3; a ++ ) tcen[ i * 3 + a ] = ( b.min[ a ] + b.max[ a ] ) * 0.5;

	}

	const top = buildBvh( subtrees.length, tb, tcen, { leafSize: 1, maxDepth: 36 } );
	const base = out.length;
	let depth = 0;

	// Emit in depth-first order. A top leaf is replaced by its level's subtree (same DFS layout); a leaf that holds several
	// items (coincident centroids past the depth limit) becomes a right-leaning chain of internal nodes over them.
	const boxOfCluster = ( c ) => [ clusterBox.subarray( c * 6, c * 6 + 3 ), clusterBox.subarray( c * 6 + 3, c * 6 + 6 ) ];

	/** items: [{ min, max, emit( d ) → node index }] */
	const emitList = ( items, d ) => {

		if ( d > depth ) depth = d;
		if ( items.length === 1 ) return items[ 0 ].emit( d );
		const mn = [ Infinity, Infinity, Infinity ], mx = [ - Infinity, - Infinity, - Infinity ];
		for ( const it of items ) for ( let a = 0; a < 3; a ++ ) { mn[ a ] = Math.min( mn[ a ], it.min[ a ] ); mx[ a ] = Math.max( mx[ a ], it.max[ a ] ); }
		const idx = out.length, rec = node( mn, mx, 0 );
		out.push( rec );
		emitList( [ items[ 0 ] ], d + 1 );               // left = idx + 1
		rec.word = emitList( items.slice( 1 ), d + 1 );
		return idx;

	};

	const clusterItem = ( c ) => { const [ mn, mx ] = boxOfCluster( c ); return { min: mn, max: mx, emit: ( d ) => { if ( d > depth ) depth = d; const idx = out.length, rec = node( mn, mx, 0 ); out.push( rec ); leafRecord( rec, c ); return idx; } }; };

	const emitSub = ( sub, i, d ) => {

		const b = sub.bvh;
		if ( d > depth ) depth = d;
		if ( b.right[ i ] < 0 ) {

			const items = [];
			for ( let k = b.first[ i ]; k < b.first[ i ] + b.count[ i ]; k ++ ) items.push( clusterItem( sub.ids[ b.order[ k ] ] ) );
			return emitList( items, d );

		}

		const idx = out.length, rec = node( b.min, b.max, i );
		out.push( rec );
		emitSub( sub, i + 1, d + 1 );
		rec.word = emitSub( sub, b.right[ i ], d + 1 );
		return idx;

	};

	const subItem = ( s ) => ( { min: s.bvh.min.subarray( 0, 3 ), max: s.bvh.max.subarray( 0, 3 ), emit: ( d ) => emitSub( s, 0, d ) } );

	const emitTop = ( i, d ) => {

		if ( top.right[ i ] < 0 ) {

			const items = [];
			for ( let k = top.first[ i ]; k < top.first[ i ] + top.count[ i ]; k ++ ) items.push( subItem( subtrees[ top.order[ k ] ] ) );
			return emitList( items, d );

		}

		const idx = out.length, rec = node( top.min.subarray( i * 3, i * 3 + 3 ), top.max.subarray( i * 3, i * 3 + 3 ), 0 );
		out.push( rec );
		emitTop( i + 1, d + 1 );
		rec.word = emitTop( top.right[ i ], d + 1 );
		return idx;

	};

	emitTop( 0, 0 );

	// LOD aggregates, children before parents (DFS order: children have larger indices)
	for ( let i = out.length - 1; i >= base; i -- ) {

		const h = out[ i ];
		if ( h.word & LEAF_BIT ) continue;
		const l = out[ i + 1 ], r = out[ h.word ];
		h.maxParentError = Math.max( l.maxParentError, r.maxParentError );
		h.minOwnError = Math.min( l.minOwnError, r.minOwnError );
		h.maxParentRadius = Math.max( l.maxParentRadius, r.maxParentRadius );
		for ( let a = 0; a < 3; a ++ ) { h.lodMin[ a ] = Math.min( l.lodMin[ a ], r.lodMin[ a ] ); h.lodMax[ a ] = Math.max( l.lodMax[ a ], r.lodMax[ a ] ); }

	}

	return { depth };

	function node( mn, mx, i ) {

		return { min: [ mn[ i * 3 ], mn[ i * 3 + 1 ], mn[ i * 3 + 2 ] ], max: [ mx[ i * 3 ], mx[ i * 3 + 1 ], mx[ i * 3 + 2 ] ], word: 0, maxParentError: 0, minOwnError: Infinity, lodMin: [ Infinity, Infinity, Infinity ], lodMax: [ - Infinity, - Infinity, - Infinity ], maxParentRadius: 0 };

	}

	function leafRecord( rec, c ) {

		const o = c * 12;
		rec.word = ( LEAF_BIT | c ) >>> 0;
		rec.maxParentError = lod[ o + 9 ];
		rec.minOwnError = lod[ o + 8 ];
		rec.maxParentRadius = lod[ o + 7 ];
		for ( let a = 0; a < 3; a ++ ) {

			rec.lodMin[ a ] = Math.min( lod[ o + a ], lod[ o + 4 + a ] );
			rec.lodMax[ a ] = Math.max( lod[ o + a ], lod[ o + 4 + a ] );

		}

	}

}

// --- TLAS ------------------------------------------------------------------------------------------------------------------

/**
 * Build the per-frame TLAS over the instances and pack it: nodes first, then one record per instance.
 *
 *   node (2 vec4): [min.xyz, word] [max.xyz, 0]; word bit 31 = leaf (low bits: instance index), else right child (left = node + 1)
 *   instance record (10 vec4) at instanceBase + i * 10: world matrix M (4 vec4, column-major), inverse(M) (4 vec4),
 *     [hierarchy root node, maxScale, minScale, mesh index], [pruneMax, pruneMin, 0, 0]
 *     maxScale is the largest column norm of M, exactly what the cull kernel uses in the cut rule; pruneMax and pruneMin bound the
 *     singular values of M (the most and the least a vector can be stretched), which the hierarchy prunes need: for orthogonal
 *     columns they are the column norms, otherwise the Frobenius norms of M and of its inverse give safe bounds; both carry a
 *     0.1 % margin so single-precision rounding at a prune boundary stays on the safe side
 *
 * @param {Object} params
 * @param {number} params.instanceCount
 * @param {Float32Array} params.instanceMatrices   16 floats per instance, column-major (object → mesh space)
 * @param {ArrayLike<number>} [params.modelMatrix]  mesh → world (identity if omitted)
 * @param {Uint32Array|null} [params.instanceMesh]  mesh index per instance (null: all mesh 0)
 * @param {Array<{ first: number, count: number, boundingBox?: { min: {x,y,z}, max: {x,y,z} } }>} params.meshes  mesh records (the set's, or one)
 * @param {Float32Array} params.meshBox            6 floats per mesh: object-space box (fallback when a record has none)
 * @param {Uint32Array} params.meshRoot             hierarchy root per mesh (AccelResult.meshRoot)
 * @param {Float32Array} [out]                      reuse a packed buffer when large enough
 * @returns {{ data: Float32Array, vec4Count: number, nodeCount: number, instanceBase: number, depth: number, worldBox: Float32Array }}
 */
export function buildInstanceTlas( params, out = null ) {

	const { instanceCount: n, instanceMatrices, meshRoot, meshBox } = params;
	const model = params.modelMatrix || IDENTITY;
	const instanceMesh = params.instanceMesh || null;
	const boxes = new Float32Array( Math.max( 1, n ) * 6 ), centroids = new Float32Array( Math.max( 1, n ) * 3 );
	const M = new Float64Array( 16 * Math.max( 1, n ) );
	for ( let i = 0; i < n; i ++ ) {

		multiply4( model, instanceMatrices, i * 16, M, i * 16 );
		const k = instanceMesh ? instanceMesh[ i ] : 0;
		const b = k * 6;
		let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = - Infinity, y1 = - Infinity, z1 = - Infinity;
		for ( let corner = 0; corner < 8; corner ++ ) {

			const x = corner & 1 ? meshBox[ b + 3 ] : meshBox[ b ], y = corner & 2 ? meshBox[ b + 4 ] : meshBox[ b + 1 ], z = corner & 4 ? meshBox[ b + 5 ] : meshBox[ b + 2 ];
			const o = i * 16;
			const wx = M[ o ] * x + M[ o + 4 ] * y + M[ o + 8 ] * z + M[ o + 12 ];
			const wy = M[ o + 1 ] * x + M[ o + 5 ] * y + M[ o + 9 ] * z + M[ o + 13 ];
			const wz = M[ o + 2 ] * x + M[ o + 6 ] * y + M[ o + 10 ] * z + M[ o + 14 ];
			if ( wx < x0 ) x0 = wx; if ( wy < y0 ) y0 = wy; if ( wz < z0 ) z0 = wz; if ( wx > x1 ) x1 = wx; if ( wy > y1 ) y1 = wy; if ( wz > z1 ) z1 = wz;

		}

		boxes[ i * 6 ] = x0; boxes[ i * 6 + 1 ] = y0; boxes[ i * 6 + 2 ] = z0; boxes[ i * 6 + 3 ] = x1; boxes[ i * 6 + 4 ] = y1; boxes[ i * 6 + 5 ] = z1;
		centroids[ i * 3 ] = ( x0 + x1 ) * 0.5; centroids[ i * 3 + 1 ] = ( y0 + y1 ) * 0.5; centroids[ i * 3 + 2 ] = ( z0 + z1 ) * 0.5;

	}

	// near-coincident instances tie their boxes, and the SAH then peels one per level: fall back to a balanced build when the stack would not hold
	let bvh = buildBvh( n, boxes, centroids, { leafSize: 1, maxDepth: 64 } );
	if ( bvh.depth >= TLAS_STACK_DEPTH - 1 ) bvh = buildBvh( n, boxes, centroids, { leafSize: 1, maxDepth: 64, balanced: true } );
	if ( bvh.depth >= TLAS_STACK_DEPTH - 1 ) throw new Error( `buildInstanceTlas: ${ n } instances need a traversal stack of ${ bvh.depth + 2 }, the kernel has ${ TLAS_STACK_DEPTH }` );
	const nodeCount = bvh.nodeCount;
	const instanceBase = nodeCount * TLAS_NODE_VEC4;
	const vec4Count = instanceBase + Math.max( 1, n ) * TLAS_INSTANCE_VEC4;
	const data = out && out.length >= vec4Count * 4 ? out : new Float32Array( vec4Count * 4 );
	const u = new Uint32Array( data.buffer, data.byteOffset, data.length );

	for ( let i = 0; i < nodeCount; i ++ ) {

		const o = i * TLAS_NODE_VEC4 * 4;
		data[ o ] = bvh.min[ i * 3 ]; data[ o + 1 ] = bvh.min[ i * 3 + 1 ]; data[ o + 2 ] = bvh.min[ i * 3 + 2 ];
		data[ o + 4 ] = bvh.max[ i * 3 ]; data[ o + 5 ] = bvh.max[ i * 3 + 1 ]; data[ o + 6 ] = bvh.max[ i * 3 + 2 ];
		u[ o + 3 ] = bvh.right[ i ] < 0 ? ( ( LEAF_BIT | ( n ? bvh.order[ bvh.first[ i ] ] : 0 ) ) >>> 0 ) : bvh.right[ i ];
		u[ o + 7 ] = n === 0 && bvh.right[ i ] < 0 ? 1 : 0; // an empty scene: the root leaf is marked empty

	}

	const inv = new Float64Array( 16 );
	for ( let i = 0; i < n; i ++ ) {

		const o = ( instanceBase + i * TLAS_INSTANCE_VEC4 ) * 4, m = i * 16;
		for ( let k = 0; k < 16; k ++ ) data[ o + k ] = M[ m + k ];
		invert4( M, m, inv );
		for ( let k = 0; k < 16; k ++ ) data[ o + 16 + k ] = inv[ k ];
		const sx = Math.hypot( M[ m ], M[ m + 1 ], M[ m + 2 ] ), sy = Math.hypot( M[ m + 4 ], M[ m + 5 ], M[ m + 6 ] ), sz = Math.hypot( M[ m + 8 ], M[ m + 9 ], M[ m + 10 ] );
		const k = instanceMesh ? instanceMesh[ i ] : 0;
		u[ o + 32 ] = meshRoot[ k ]; data[ o + 33 ] = Math.max( sx, sy, sz ); data[ o + 34 ] = Math.min( sx, sy, sz ); u[ o + 35 ] = k;
		// bounds on how much M can stretch or shrink a vector: the column norms when the columns are orthogonal (they are then the singular
		// values), else the Frobenius norms of M and of its inverse (sigma_max <= |M|_F, sigma_min >= 1 / |M^-1|_F)
		const d01 = M[ m ] * M[ m + 4 ] + M[ m + 1 ] * M[ m + 5 ] + M[ m + 2 ] * M[ m + 6 ];
		const d02 = M[ m ] * M[ m + 8 ] + M[ m + 1 ] * M[ m + 9 ] + M[ m + 2 ] * M[ m + 10 ];
		const d12 = M[ m + 4 ] * M[ m + 8 ] + M[ m + 5 ] * M[ m + 9 ] + M[ m + 6 ] * M[ m + 10 ];
		const orthogonal = Math.abs( d01 ) <= 1e-4 * sx * sy && Math.abs( d02 ) <= 1e-4 * sx * sz && Math.abs( d12 ) <= 1e-4 * sy * sz;
		let pruneMax, pruneMin;
		if ( orthogonal ) { pruneMax = Math.max( sx, sy, sz ); pruneMin = Math.min( sx, sy, sz ); }
		else {

			pruneMax = Math.sqrt( sx * sx + sy * sy + sz * sz );
			const invF = Math.sqrt( inv[ 0 ] * inv[ 0 ] + inv[ 1 ] * inv[ 1 ] + inv[ 2 ] * inv[ 2 ] + inv[ 4 ] * inv[ 4 ] + inv[ 5 ] * inv[ 5 ] + inv[ 6 ] * inv[ 6 ] + inv[ 8 ] * inv[ 8 ] + inv[ 9 ] * inv[ 9 ] + inv[ 10 ] * inv[ 10 ] );
			pruneMin = invF > 0 ? 1 / invF : 0;

		}

		data[ o + 36 ] = pruneMax * 1.001; data[ o + 37 ] = pruneMin * 0.999; data[ o + 38 ] = 0; data[ o + 39 ] = 0;

	}

	const worldBox = n ? new Float32Array( [ bvh.min[ 0 ], bvh.min[ 1 ], bvh.min[ 2 ], bvh.max[ 0 ], bvh.max[ 1 ], bvh.max[ 2 ] ] ) : new Float32Array( 6 );
	return { data, vec4Count, nodeCount, instanceBase, depth: bvh.depth, worldBox, instanceCount: n };

}

const IDENTITY = new Float32Array( [ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 ] );

/** out[oo..] = a * b[bo..] (column-major 4x4). */
export function multiply4( a, b, bo, out, oo = 0 ) {

	for ( let c = 0; c < 4; c ++ ) for ( let r = 0; r < 4; r ++ ) {

		out[ oo + c * 4 + r ] = a[ r ] * b[ bo + c * 4 ] + a[ 4 + r ] * b[ bo + c * 4 + 1 ] + a[ 8 + r ] * b[ bo + c * 4 + 2 ] + a[ 12 + r ] * b[ bo + c * 4 + 3 ];

	}

	return out;

}

/** General 4x4 inverse (column-major), as three.js Matrix4.invert; a singular matrix gives zeros. */
export function invert4( m, mo, out ) {

	const n11 = m[ mo ], n21 = m[ mo + 1 ], n31 = m[ mo + 2 ], n41 = m[ mo + 3 ];
	const n12 = m[ mo + 4 ], n22 = m[ mo + 5 ], n32 = m[ mo + 6 ], n42 = m[ mo + 7 ];
	const n13 = m[ mo + 8 ], n23 = m[ mo + 9 ], n33 = m[ mo + 10 ], n43 = m[ mo + 11 ];
	const n14 = m[ mo + 12 ], n24 = m[ mo + 13 ], n34 = m[ mo + 14 ], n44 = m[ mo + 15 ];
	const t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
	const t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
	const t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
	const t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;
	const det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
	if ( det === 0 ) { out.fill( 0 ); return out; }
	const d = 1 / det;
	out[ 0 ] = t11 * d;
	out[ 1 ] = ( n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 - n21 * n33 * n44 ) * d;
	out[ 2 ] = ( n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44 ) * d;
	out[ 3 ] = ( n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43 ) * d;
	out[ 4 ] = t12 * d;
	out[ 5 ] = ( n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44 ) * d;
	out[ 6 ] = ( n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44 ) * d;
	out[ 7 ] = ( n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43 ) * d;
	out[ 8 ] = t13 * d;
	out[ 9 ] = ( n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44 ) * d;
	out[ 10 ] = ( n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44 ) * d;
	out[ 11 ] = ( n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43 ) * d;
	out[ 12 ] = t14 * d;
	out[ 13 ] = ( n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34 ) * d;
	out[ 14 ] = ( n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34 ) * d;
	out[ 15 ] = ( n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33 ) * d;
	return out;

}

/** Object-space box of a set or of each mesh of a merged set: 6 floats per mesh. */
export function meshBoxes( set ) {

	const meshes = set.meshes || [ { first: 0, count: set.meshletCount, boundingBox: set.boundingBox } ];
	const out = new Float32Array( meshes.length * 6 );
	for ( let k = 0; k < meshes.length; k ++ ) {

		const r = meshes[ k ], o = k * 6;
		if ( r.boundingBox && r.boundingBox.min ) {

			out[ o ] = r.boundingBox.min.x; out[ o + 1 ] = r.boundingBox.min.y; out[ o + 2 ] = r.boundingBox.min.z;
			out[ o + 3 ] = r.boundingBox.max.x; out[ o + 4 ] = r.boundingBox.max.y; out[ o + 5 ] = r.boundingBox.max.z;

		} else {

			// from the clusters' bounding spheres
			let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = - Infinity, y1 = - Infinity, z1 = - Infinity;
			for ( let c = r.first; c < r.first + r.count; c ++ ) {

				const b = c * 12, rad = set.bounds[ b + 3 ];
				x0 = Math.min( x0, set.bounds[ b ] - rad ); y0 = Math.min( y0, set.bounds[ b + 1 ] - rad ); z0 = Math.min( z0, set.bounds[ b + 2 ] - rad );
				x1 = Math.max( x1, set.bounds[ b ] + rad ); y1 = Math.max( y1, set.bounds[ b + 1 ] + rad ); z1 = Math.max( z1, set.bounds[ b + 2 ] + rad );

			}

			out[ o ] = x0; out[ o + 1 ] = y0; out[ o + 2 ] = z0; out[ o + 3 ] = x1; out[ o + 4 ] = y1; out[ o + 5 ] = z1;

		}

	}

	return out;

}

function now() { return ( typeof performance !== 'undefined' && performance.now ) ? performance.now() : Date.now(); }
