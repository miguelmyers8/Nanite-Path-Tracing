/**
 * A small binary BVH builder over items with bounding boxes (binned SAH with a
 * median fallback), shared by the per-cluster triangle BVHs, the per-mesh
 * cluster hierarchy and the per-frame instance TLAS.
 *
 * Nodes come out in depth-first order: the left child of node i is node
 * i + 1 and the right child is `right[i]`; a leaf has `right[i] === -1` and
 * covers `order[first[i] .. first[i] + count[i])`. The GPU kernels and the
 * CPU reference traverse this layout with an explicit stack.
 *
 * No three.js dependency.
 *
 * @module bvh
 */

const BINS = 12;

/**
 * @typedef {Object} BvhResult
 * @property {number} nodeCount
 * @property {Float32Array} min       3 floats per node
 * @property {Float32Array} max       3 floats per node
 * @property {Int32Array} right       right child index, -1 for a leaf
 * @property {Uint32Array} first      leaf: first index into `order`
 * @property {Uint32Array} count      leaf: item count (0 for an internal node)
 * @property {Uint32Array} order      item permutation
 * @property {number} depth           deepest leaf (root = 0)
 */

/**
 * Build a BVH.
 *
 * @param {number} count          item count
 * @param {Float32Array} boxes    6 floats per item: min xyz, max xyz
 * @param {Float32Array} centroids 3 floats per item
 * @param {Object} [options]
 * @param {number} [options.leafSize=1]   maximum items per leaf
 * @param {number} [options.maxDepth=40]  below this depth the builder stops splitting (bigger leaves instead)
 * @returns {BvhResult}
 */
export function buildBvh( count, boxes, centroids, options = {} ) {

	const leafSize = Math.max( 1, options.leafSize ?? 1 );
	const maxDepth = options.maxDepth ?? 40;
	const order = new Uint32Array( count );
	for ( let i = 0; i < count; i ++ ) order[ i ] = i;

	// worst case 2n - 1 nodes
	const cap = Math.max( 1, 2 * count );
	const min = new Float32Array( cap * 3 ), max = new Float32Array( cap * 3 );
	const right = new Int32Array( cap ), first = new Uint32Array( cap ), cnt = new Uint32Array( cap );
	let nodeCount = 0, depthMax = 0;

	const binCount = new Uint32Array( BINS ), binMin = new Float64Array( BINS * 3 ), binMax = new Float64Array( BINS * 3 );
	const leftArea = new Float64Array( BINS ), leftCount = new Uint32Array( BINS );

	const emit = ( f, n, depth ) => {

		const idx = nodeCount ++;
		if ( depth > depthMax ) depthMax = depth;

		// bounds of the items and of their centroids
		let bx0 = Infinity, by0 = Infinity, bz0 = Infinity, bx1 = - Infinity, by1 = - Infinity, bz1 = - Infinity;
		let cx0 = Infinity, cy0 = Infinity, cz0 = Infinity, cx1 = - Infinity, cy1 = - Infinity, cz1 = - Infinity;
		for ( let k = f; k < f + n; k ++ ) {

			const i = order[ k ], b = i * 6, c = i * 3;
			if ( boxes[ b ] < bx0 ) bx0 = boxes[ b ]; if ( boxes[ b + 1 ] < by0 ) by0 = boxes[ b + 1 ]; if ( boxes[ b + 2 ] < bz0 ) bz0 = boxes[ b + 2 ];
			if ( boxes[ b + 3 ] > bx1 ) bx1 = boxes[ b + 3 ]; if ( boxes[ b + 4 ] > by1 ) by1 = boxes[ b + 4 ]; if ( boxes[ b + 5 ] > bz1 ) bz1 = boxes[ b + 5 ];
			if ( centroids[ c ] < cx0 ) cx0 = centroids[ c ]; if ( centroids[ c + 1 ] < cy0 ) cy0 = centroids[ c + 1 ]; if ( centroids[ c + 2 ] < cz0 ) cz0 = centroids[ c + 2 ];
			if ( centroids[ c ] > cx1 ) cx1 = centroids[ c ]; if ( centroids[ c + 1 ] > cy1 ) cy1 = centroids[ c + 1 ]; if ( centroids[ c + 2 ] > cz1 ) cz1 = centroids[ c + 2 ];

		}

		min[ idx * 3 ] = bx0; min[ idx * 3 + 1 ] = by0; min[ idx * 3 + 2 ] = bz0;
		max[ idx * 3 ] = bx1; max[ idx * 3 + 1 ] = by1; max[ idx * 3 + 2 ] = bz1;

		const ex = cx1 - cx0, ey = cy1 - cy0, ez = cz1 - cz0;
		const axis = ex >= ey && ex >= ez ? 0 : ey >= ez ? 1 : 2;
		const extent = axis === 0 ? ex : axis === 1 ? ey : ez;

		if ( n <= leafSize || depth >= maxDepth ) {

			right[ idx ] = - 1; first[ idx ] = f; cnt[ idx ] = n;
			return idx;

		}

		// binned SAH along the longest centroid axis (coincident centroids: split at the median so leaves stay small)
		const c0 = axis === 0 ? cx0 : axis === 1 ? cy0 : cz0;
		const scale = extent > 0 ? BINS / extent : 0;
		binCount.fill( 0 ); binMin.fill( Infinity ); binMax.fill( - Infinity );
		if ( extent > 0 ) for ( let k = f; k < f + n; k ++ ) {

			const i = order[ k ];
			let bin = ( ( centroids[ i * 3 + axis ] - c0 ) * scale ) | 0;
			if ( bin >= BINS ) bin = BINS - 1;
			binCount[ bin ] ++;
			const b = i * 6, o = bin * 3;
			for ( let a = 0; a < 3; a ++ ) { if ( boxes[ b + a ] < binMin[ o + a ] ) binMin[ o + a ] = boxes[ b + a ]; if ( boxes[ b + 3 + a ] > binMax[ o + a ] ) binMax[ o + a ] = boxes[ b + 3 + a ]; }

		}

		// sweep: left prefix areas, then right suffix
		let lx0 = Infinity, ly0 = Infinity, lz0 = Infinity, lx1 = - Infinity, ly1 = - Infinity, lz1 = - Infinity, lc = 0;
		for ( let b = 0; b < BINS - 1; b ++ ) {

			const o = b * 3;
			if ( binCount[ b ] ) { lx0 = Math.min( lx0, binMin[ o ] ); ly0 = Math.min( ly0, binMin[ o + 1 ] ); lz0 = Math.min( lz0, binMin[ o + 2 ] ); lx1 = Math.max( lx1, binMax[ o ] ); ly1 = Math.max( ly1, binMax[ o + 1 ] ); lz1 = Math.max( lz1, binMax[ o + 2 ] ); lc += binCount[ b ]; }
			leftCount[ b ] = lc; leftArea[ b ] = lc ? area( lx0, ly0, lz0, lx1, ly1, lz1 ) : 0;

		}

		let bestCost = Infinity, bestBin = - 1;
		let rx0 = Infinity, ry0 = Infinity, rz0 = Infinity, rx1 = - Infinity, ry1 = - Infinity, rz1 = - Infinity, rc = 0;
		for ( let b = BINS - 1; b >= 1; b -- ) {

			const o = b * 3;
			if ( binCount[ b ] ) { rx0 = Math.min( rx0, binMin[ o ] ); ry0 = Math.min( ry0, binMin[ o + 1 ] ); rz0 = Math.min( rz0, binMin[ o + 2 ] ); rx1 = Math.max( rx1, binMax[ o ] ); ry1 = Math.max( ry1, binMax[ o + 1 ] ); rz1 = Math.max( rz1, binMax[ o + 2 ] ); rc += binCount[ b ]; }
			const lcount = leftCount[ b - 1 ];
			if ( lcount === 0 || rc === 0 ) continue;
			const cost = lcount * leftArea[ b - 1 ] + rc * area( rx0, ry0, rz0, rx1, ry1, rz1 );
			if ( cost < bestCost ) { bestCost = cost; bestBin = b; }

		}

		// partition `order[f .. f + n)` by the chosen bin boundary (or at the median when no bin split separates the items)
		let k;
		if ( bestBin >= 0 ) {

			let lo = f, hi = f + n - 1;
			while ( lo <= hi ) {

				const i = order[ lo ];
				let bin = ( ( centroids[ i * 3 + axis ] - c0 ) * scale ) | 0;
				if ( bin >= BINS ) bin = BINS - 1;
				if ( bin < bestBin ) lo ++; else { order[ lo ] = order[ hi ]; order[ hi ] = i; hi --; }

			}

			k = lo - f;

		}

		if ( bestBin < 0 || k === 0 || k === n ) {

			const sub = Array.from( order.subarray( f, f + n ) ).sort( ( a, b ) => centroids[ a * 3 + axis ] - centroids[ b * 3 + axis ] );
			order.set( sub, f );
			k = n >> 1;

		}

		emit( f, k, depth + 1 );            // left child is idx + 1
		right[ idx ] = emit( f + k, n - k, depth + 1 );
		first[ idx ] = f; cnt[ idx ] = 0;
		return idx;

	};

	if ( count === 0 ) {

		nodeCount = 1; right[ 0 ] = - 1; first[ 0 ] = 0; cnt[ 0 ] = 0;
		min.fill( 0, 0, 3 ); max.fill( 0, 0, 3 );

	} else emit( 0, count, 0 );

	return {
		nodeCount,
		min: min.subarray( 0, nodeCount * 3 ), max: max.subarray( 0, nodeCount * 3 ),
		right: right.subarray( 0, nodeCount ), first: first.subarray( 0, nodeCount ), count: cnt.subarray( 0, nodeCount ),
		order, depth: depthMax,
	};

}

function area( x0, y0, z0, x1, y1, z1 ) {

	const dx = Math.max( 0, x1 - x0 ), dy = Math.max( 0, y1 - y0 ), dz = Math.max( 0, z1 - z0 );
	return 2 * ( dx * dy + dy * dz + dz * dx );

}
