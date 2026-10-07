import { test } from 'node:test';
import assert from 'node:assert/strict';
import { lodSelected } from 'nanite/meshlets/core.js';
import {
	buildBvh, buildAccel, buildInstanceTlas, meshBoxes, traceRay, traceRayBruteForce, cameraRay,
	HIER_VEC4, CBVH_VEC4, TLAS_INSTANCE_VEC4, LEAF_BIT, STACK_HIERARCHY, STACK_CLUSTER,
} from '../src/core/index.js';
import { makeTorus, makeSphere, makeLodSet, makePlainSet, compose, lookAt } from './helpers.mjs';

// --- generic builder --------------------------------------------------------------------------------------------------

test( 'buildBvh: permutation, leaf sizes, nested boxes, DFS layout', () => {

	let seed = 7; const rnd = () => ( seed = ( seed * 1664525 + 1013904223 ) >>> 0 ) / 4294967296;
	const n = 500, boxes = new Float32Array( n * 6 ), centroids = new Float32Array( n * 3 );
	for ( let i = 0; i < n; i ++ ) {

		const x = rnd() * 10, y = rnd() * 10, z = rnd() * 10, r = 0.05 + rnd() * 0.3;
		boxes.set( [ x - r, y - r, z - r, x + r, y + r, z + r ], i * 6 ); centroids.set( [ x, y, z ], i * 3 );

	}

	for ( const leafSize of [ 1, 4 ] ) {

		const b = buildBvh( n, boxes, centroids, { leafSize } );
		const seen = new Uint8Array( n );
		let leaves = 0, items = 0, depth = 0;
		const walk = ( i, d ) => {

			depth = Math.max( depth, d );
			if ( b.right[ i ] < 0 ) {

				leaves ++; assert.ok( b.count[ i ] >= 1 && b.count[ i ] <= leafSize, `leaf size ${ b.count[ i ] }` );
				for ( let k = b.first[ i ]; k < b.first[ i ] + b.count[ i ]; k ++ ) {

					const it = b.order[ k ]; seen[ it ] ++; items ++;
					for ( let a = 0; a < 3; a ++ ) { assert.ok( boxes[ it * 6 + a ] >= b.min[ i * 3 + a ] - 1e-6 ); assert.ok( boxes[ it * 6 + 3 + a ] <= b.max[ i * 3 + a ] + 1e-6 ); }

				}

				return i + 1;

			}

			const l = i + 1, r = b.right[ i ];
			assert.ok( r > l && r < b.nodeCount );
			for ( const c of [ l, r ] ) for ( let a = 0; a < 3; a ++ ) { assert.ok( b.min[ c * 3 + a ] >= b.min[ i * 3 + a ] - 1e-6 ); assert.ok( b.max[ c * 3 + a ] <= b.max[ i * 3 + a ] + 1e-6 ); }
			const next = walk( l, d + 1 );
			assert.equal( next, r, 'right child follows the left subtree in DFS order' );
			return walk( r, d + 1 );

		};

		assert.equal( walk( 0, 0 ), b.nodeCount );
		assert.equal( items, n ); assert.ok( seen.every( ( s ) => s === 1 ) );
		assert.equal( depth, b.depth );
		assert.ok( leaves <= n && leaves >= Math.ceil( n / leafSize ) );

	}

} );

// --- accel structure --------------------------------------------------------------------------------------------------

test( 'buildAccel: cluster BVHs cover every triangle once, hierarchy covers every cluster once, aggregates nest', () => {

	const set = makeLodSet( makeTorus() );
	const accel = buildAccel( set );
	const A = accel.data, AU = new Uint32Array( A.buffer ), L = accel.layout;
	assert.ok( set.levels.length > 3, `levels ${ set.levels.length }` );
	assert.ok( accel.maxHierarchyDepth < STACK_HIERARCHY - 1 && accel.maxClusterDepth < STACK_CLUSTER - 1 );

	// cluster BVHs
	for ( let c = 0; c < set.meshletCount; c ++ ) {

		const t = ( L.tableBase + c ) * 4, nodeBase = AU[ t ], orderBase = AU[ t + 1 ], nodeCount = AU[ t + 2 ];
		const tc = set.meshlets[ c * 4 + 3 ];
		const seen = new Uint8Array( tc );
		for ( let i = 0; i < nodeCount; i ++ ) {

			const o = ( L.clusterBvhBase + ( nodeBase + i ) * CBVH_VEC4 ) * 4, count = AU[ o + 7 ];
			if ( count === 0 ) { assert.ok( AU[ o + 3 ] > nodeBase + i && AU[ o + 3 ] < nodeBase + nodeCount ); continue; }
			for ( let k = 0; k < count; k ++ ) seen[ accel.order[ orderBase + AU[ o + 3 ] + k ] ] ++;

		}

		assert.ok( seen.every( ( s ) => s === 1 ), `cluster ${ c }: every triangle in exactly one leaf` );

	}

	// hierarchy
	const seen = new Uint8Array( set.meshletCount );
	for ( let i = 0; i < L.hierarchyNodeCount; i ++ ) {

		const o = ( L.hierarchyBase + i * HIER_VEC4 ) * 4, word = AU[ o + 3 ];
		if ( word & LEAF_BIT ) { seen[ word & 0x7fffffff ] ++; continue; }
		const l = i + 1, r = word;
		assert.ok( r > l && r < L.hierarchyNodeCount );
		for ( const c of [ l, r ] ) {

			const co = ( L.hierarchyBase + c * HIER_VEC4 ) * 4;
			for ( let a = 0; a < 3; a ++ ) {

				assert.ok( A[ co + a ] >= A[ o + a ] - 1e-6 && A[ co + 4 + a ] <= A[ o + 4 + a ] + 1e-6, 'geometry boxes nest' );
				assert.ok( A[ co + 8 + a ] >= A[ o + 8 + a ] - 1e-6 && A[ co + 12 + a ] <= A[ o + 12 + a ] + 1e-6, 'lod boxes nest' );

			}

			assert.ok( A[ co + 7 ] <= A[ o + 7 ], 'maxParentError is a max' );
			assert.ok( A[ co + 11 ] >= A[ o + 11 ], 'minOwnError is a min' );
			assert.ok( A[ co + 15 ] <= A[ o + 15 ], 'maxParentRadius is a max' );

		}

	}

	assert.ok( seen.every( ( s ) => s === 1 ), 'every cluster is exactly one hierarchy leaf' );
	// a root cluster's parent error is +inf and it is the top of the aggregate
	assert.equal( A[ ( L.hierarchyBase + accel.meshRoot[ 0 ] * HIER_VEC4 ) * 4 + 7 ], Infinity );
	assert.equal( A[ ( L.hierarchyBase + accel.meshRoot[ 0 ] * HIER_VEC4 ) * 4 + 11 ], 0 );

} );

// --- tracing ------------------------------------------------------------------------------------------------------------

function sceneOf( set, instances ) {

	const accel = buildAccel( set );
	const matrices = new Float32Array( instances.length * 16 );
	instances.forEach( ( m, i ) => matrices.set( m, i * 16 ) );
	const tlas = buildInstanceTlas( { instanceCount: instances.length, instanceMatrices: matrices, meshes: set.meshes || [ { first: 0, count: set.meshletCount } ], meshBox: meshBoxes( set ), meshRoot: accel.meshRoot } );
	return { set, accel, tlas };

}

function sameHit( a, b, msg ) {

	if ( ! a || ! b ) { assert.equal( a, b, msg ); return; }
	assert.equal( a.instance, b.instance, msg + ' instance' ); assert.equal( a.cluster, b.cluster, msg + ' cluster' ); assert.equal( a.triangle, b.triangle, msg + ' triangle' );
	assert.ok( Math.abs( a.t - b.t ) <= 1e-6 * Math.max( 1, a.t ), msg + ' t' );

}

function rayGrid( ctx, camWorld, lod, n = 24, fov = 50 * Math.PI / 180 ) {

	const W = 64, H = 48, stats = { nodes: 0, clusters: 0, selected: 0, triangles: 0 };
	let hits = 0, compared = 0;
	const clusters = new Set();
	for ( let j = 0; j < n; j ++ ) for ( let i = 0; i < n; i ++ ) {

		const { origin, direction } = cameraRay( camWorld, fov, W, H, i * W / n + 0.37, j * H / n + 0.61 );
		const a = traceRay( ctx, origin, direction, lod, { stats } ), b = traceRayBruteForce( ctx, origin, direction, lod );
		sameHit( a, b, `ray ${ i },${ j }` );
		compared ++;
		if ( a ) { hits ++; clusters.add( a.cluster ); }

	}

	return { hits, compared, stats, clusters };

}

test( 'traceRay equals the brute force over the cut at several distances and thresholds (one instance)', () => {

	const set = makeLodSet( makeTorus() );
	const ctx = sceneOf( set, [ compose( 0, 0, 0 ) ] );
	const levels = new Set();
	for ( const dist of [ 3.5, 8, 30, 120 ] ) for ( const threshold of [ 0.5, 2, 8 ] ) {

		const eye = [ dist * 0.6, dist * 0.5, dist * 0.62 ];
		const cam = lookAt( eye, [ 0, 0, 0 ] );
		// a field of view that keeps the torus (radius 2.6) filling the frame, and the matching pixel scale, as a camera would
		const fov = Math.min( 50 * Math.PI / 180, 2 * Math.atan( 3.2 / dist ) ), pixelScale = 1080 / ( 2 * Math.tan( fov / 2 ) );
		const lod = { cameraPosition: eye, pixelScale, threshold, near: 0.1 };
		const r = rayGrid( ctx, cam, lod, 24, fov );
		assert.ok( r.hits > r.compared * 0.1, `hits at ${ dist } / ${ threshold }: ${ r.hits }` );
		for ( const c of r.clusters ) levels.add( set.levelOf[ c ] );
		// the hierarchy must prune: far fewer clusters tested than the set has
		assert.ok( r.stats.clusters < set.meshletCount * r.compared * 0.25, `pruning ${ r.stats.clusters } tests over ${ r.compared } rays, ${ set.meshletCount } clusters` );

	}

	assert.ok( levels.size >= 3, `the camera distances reach several levels: ${ [ ...levels ].join( ',' ) }` );

} );

test( 'traceRay: a scene of two meshes, many instances, non-uniform scale, any-hit, forced level', () => {

	const torus = makeLodSet( makeTorus( 24, 48 ) ), sphere = makeLodSet( makeSphere( 48, 32, 1.2 ) );
	// a merged set by hand (mergeMeshletSets needs three.js): clusters, vertices and LOD concatenated
	const sets = [ torus, sphere ];
	const n = torus.meshletCount + sphere.meshletCount, vcount = torus.vertexCount + sphere.vertexCount;
	const meshlets = new Uint32Array( n * 4 ), vertices = new Uint32Array( torus.vertices.length + sphere.vertices.length );
	const triangles = new Uint8Array( torus.triangles.length + sphere.triangles.length ), positions = new Float32Array( vcount * 3 );
	const bounds = new Float32Array( n * 12 ), lod = new Float32Array( n * 12 ), levelOf = new Uint8Array( n ), firstTriangle = new Uint32Array( n + 1 );
	const meshes = [];
	let mo = 0, vo = 0, eo = 0, to = 0, tcount = 0;
	for ( const s of sets ) {

		for ( let i = 0; i < s.meshletCount; i ++ ) {

			meshlets[ ( mo + i ) * 4 ] = s.meshlets[ i * 4 ] + eo; meshlets[ ( mo + i ) * 4 + 1 ] = s.meshlets[ i * 4 + 1 ] + to;
			meshlets[ ( mo + i ) * 4 + 2 ] = s.meshlets[ i * 4 + 2 ]; meshlets[ ( mo + i ) * 4 + 3 ] = s.meshlets[ i * 4 + 3 ];
			firstTriangle[ mo + i ] = tcount; tcount += s.meshlets[ i * 4 + 3 ];

		}

		for ( let i = 0; i < s.vertices.length; i ++ ) vertices[ eo + i ] = s.vertices[ i ] + vo;
		triangles.set( s.triangles, to ); positions.set( s.positions, vo * 3 ); bounds.set( s.bounds, mo * 12 ); lod.set( s.lod, mo * 12 ); levelOf.set( s.levelOf, mo );
		meshes.push( { first: mo, count: s.meshletCount, boundingBox: s.boundingBox } );
		mo += s.meshletCount; vo += s.vertexCount; eo += s.vertices.length; to += s.triangles.length;

	}

	firstTriangle[ n ] = tcount;
	const merged = { isMeshletSet: true, merged: true, meshes, meshletCount: n, meshlets, vertices, triangles, firstTriangle, positions, vertexCount: vcount, triangleCount: tcount, bounds, lod, levelOf, options: torus.options };

	const instances = [ compose( 0, 0, 0 ), compose( 5, 0.5, - 2, 0.7, 1.5 ), compose( - 4, 1, 3, 2.1, 0.8, 1.6, 0.5 ), compose( 2, - 1, 6, 0.3, 1 ), compose( - 6, 0, - 5, 1.2, 2, 0.7, 1.3 ) ];
	const instanceMesh = new Uint32Array( [ 0, 1, 0, 1, 1 ] );
	const accel = buildAccel( merged );
	assert.equal( accel.meshRoot.length, 2 );
	const matrices = new Float32Array( instances.length * 16 ); instances.forEach( ( m, i ) => matrices.set( m, i * 16 ) );
	const tlas = buildInstanceTlas( { instanceCount: instances.length, instanceMatrices: matrices, instanceMesh, meshes, meshBox: meshBoxes( merged ), meshRoot: accel.meshRoot } );
	const ctx = { set: merged, accel, tlas };
	const pixelScale = 720 / ( 2 * Math.tan( 25 * Math.PI / 180 ) );

	const seenInstances = new Set();
	for ( const [ eye, threshold ] of [ [ [ 8, 6, 14 ], 1 ], [ [ - 12, 3, 2 ], 4 ], [ [ 1, 20, 1 ], 0.5 ], [ [ 40, 10, 40 ], 2 ] ] ) {

		const cam = lookAt( eye, [ 0, 0, 0 ] );
		const lod = { cameraPosition: eye, pixelScale, threshold, near: 0.1 };
		const r = rayGrid( ctx, cam, lod, 20 );
		assert.ok( r.hits > 0 );
		for ( let j = 0; j < 20; j ++ ) for ( let i = 0; i < 20; i ++ ) {

			const { origin, direction } = cameraRay( cam, 50 * Math.PI / 180, 64, 48, i * 3.2, j * 2.4 );
			const a = traceRay( ctx, origin, direction, lod ), any = traceRay( ctx, origin, direction, lod, { anyHit: true } );
			assert.equal( !! a, !! any, 'any-hit finds a hit exactly when closest-hit does' );
			if ( a ) { seenInstances.add( a.instance ); assert.ok( any.t >= a.t - 1e-9 ); }
			// the hit cluster is in the cut by Nanite's own rule
			if ( a ) {

				const ro = ( tlas.instanceBase + a.instance * TLAS_INSTANCE_VEC4 ) * 4;
				const M = tlas.data.subarray( ro, ro + 16 ), maxScale = tlas.data[ ro + 33 ];
				assert.ok( lodSelected( merged.lod, a.cluster * 12, M, maxScale, eye[ 0 ], eye[ 1 ], eye[ 2 ], pixelScale, threshold, 0.1 ) );

			}

		}

	}

	assert.ok( seenInstances.size >= 4, `instances hit: ${ [ ...seenInstances ].join( ',' ) }` );

	// forced level: every hit cluster has that level, and the brute force agrees
	for ( const forceLevel of [ 0, 2 ] ) {

		const eye = [ 8, 6, 14 ], cam = lookAt( eye, [ 0, 0, 0 ] );
		const lod = { cameraPosition: eye, pixelScale, threshold: 1, near: 0.1, forceLevel };
		const r = rayGrid( ctx, cam, lod, 12 );
		assert.ok( r.hits > 0 );
		for ( const c of r.clusters ) assert.equal( merged.levelOf[ c ], forceLevel );

	}

} );

test( 'traceRay: a closed surface has no cracks at any cut (every ray through the sphere hits)', () => {

	const set = makeLodSet( makeSphere( 96, 64, 1 ) );
	const ctx = sceneOf( set, [ compose( 0, 0, 0 ) ] );
	const pixelScale = 1080 / ( 2 * Math.tan( 25 * Math.PI / 180 ) );
	let seed = 3; const rnd = () => ( seed = ( seed * 1664525 + 1013904223 ) >>> 0 ) / 4294967296;
	const levels = new Set();
	for ( const dist of [ 2.5, 6, 20, 80, 400 ] ) {

		const eye = [ dist, dist * 0.3, dist * 0.8 ];
		const lod = { cameraPosition: eye, pixelScale, threshold: 1, near: 0.1 };
		for ( let k = 0; k < 300; k ++ ) {

			// a direction towards a random point well inside the sphere (so the ray must cross the surface, whatever the cut's error)
			const px = ( rnd() - 0.5 ) * 0.8, py = ( rnd() - 0.5 ) * 0.8, pz = ( rnd() - 0.5 ) * 0.8;
			let dx = px - eye[ 0 ], dy = py - eye[ 1 ], dz = pz - eye[ 2 ]; const l = Math.hypot( dx, dy, dz ); dx /= l; dy /= l; dz /= l;
			const h = traceRay( ctx, eye, [ dx, dy, dz ], lod );
			assert.ok( h, `ray ${ k } from ${ dist } must hit the sphere` );
			assert.ok( h.t < l, 'the hit is in front of the inner point' );
			levels.add( set.levelOf[ h.cluster ] );

		}

	}

	assert.ok( levels.size >= 3, `levels seen ${ [ ...levels ].join( ',' ) }` );

} );

test( 'traceRay: a set without a DAG traces every cluster; lodTest off ignores the rule', () => {

	const set = makePlainSet( makeTorus( 16, 32 ) );
	const ctx = sceneOf( set, [ compose( 0, 0, 0 ), compose( 0, 0, - 5, 0, 0.5 ) ] );
	const eye = [ 0, 6, 8 ], cam = lookAt( eye, [ 0, 0, - 2 ] );
	const lod = { cameraPosition: eye, pixelScale: 1000, threshold: 1, near: 0.1 };
	const r = rayGrid( ctx, cam, lod, 16 );
	assert.ok( r.hits > 0 );
	const r2 = rayGrid( ctx, cam, { ...lod, lodTest: false }, 16 );
	assert.equal( r2.hits, r.hits );

} );
