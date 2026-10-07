import { buildLodDag, MeshletSet, weldByPosition } from 'nanite/meshlets/core.js';

export function makeTorus( radialSegments = 48, tubularSegments = 96, R = 2, r = 0.6 ) {

	const positions = [], indices = [];
	for ( let j = 0; j <= radialSegments; j ++ ) for ( let i = 0; i <= tubularSegments; i ++ ) {

		const u = i / tubularSegments * Math.PI * 2, v = j / radialSegments * Math.PI * 2;
		positions.push( ( R + r * Math.cos( v ) ) * Math.cos( u ), ( R + r * Math.cos( v ) ) * Math.sin( u ), r * Math.sin( v ) );

	}

	for ( let j = 1; j <= radialSegments; j ++ ) for ( let i = 1; i <= tubularSegments; i ++ ) {

		const a = ( tubularSegments + 1 ) * j + i - 1, b = ( tubularSegments + 1 ) * ( j - 1 ) + i - 1, c = ( tubularSegments + 1 ) * ( j - 1 ) + i, d = ( tubularSegments + 1 ) * j + i;
		indices.push( a, b, d, b, c, d );

	}

	const w = weldByPosition( new Float32Array( positions ), new Uint32Array( indices ) );
	return { positions: new Float32Array( positions ), indices: w.indices };

}

export function makeSphere( widthSegments = 64, heightSegments = 48, radius = 1 ) {

	const positions = [], indices = [];
	for ( let j = 0; j <= heightSegments; j ++ ) {

		const v = j / heightSegments;
		for ( let i = 0; i <= widthSegments; i ++ ) {

			const u = i / widthSegments;
			positions.push( - radius * Math.cos( u * Math.PI * 2 ) * Math.sin( v * Math.PI ), radius * Math.cos( v * Math.PI ), radius * Math.sin( u * Math.PI * 2 ) * Math.sin( v * Math.PI ) );

		}

	}

	for ( let j = 0; j < heightSegments; j ++ ) for ( let i = 0; i < widthSegments; i ++ ) {

		const a = j * ( widthSegments + 1 ) + i + 1, b = j * ( widthSegments + 1 ) + i, c = ( j + 1 ) * ( widthSegments + 1 ) + i, d = ( j + 1 ) * ( widthSegments + 1 ) + i + 1;
		if ( j !== 0 ) indices.push( a, b, d );
		if ( j !== heightSegments - 1 ) indices.push( b, c, d );

	}

	const w = weldByPosition( new Float32Array( positions ), new Uint32Array( indices ) );
	return { positions: new Float32Array( positions ), indices: w.indices };

}

/** A LOD set like buildLodMeshletSetFromGeometry makes, without three.js. */
export function makeLodSet( { positions, indices }, options = {} ) {

	const dag = buildLodDag( indices, positions, { maxTriangles: 64, maxVertices: 64, ...options } );
	const set = new MeshletSet( dag.build, positions, dag.bounds );
	set.lod = dag.lod; set.levelOf = dag.levelOf; set.levels = dag.levels; set.groups = dag.groups; set.dag = dag;
	const box = boxOf( positions );
	set.boundingBox = { min: { x: box[ 0 ], y: box[ 1 ], z: box[ 2 ] }, max: { x: box[ 3 ], y: box[ 4 ], z: box[ 5 ] } };
	return set;

}

export function makePlainSet( { positions, indices }, options = {} ) {

	const set = MeshletSet.fromIndexedTriangles( indices, positions, { maxTriangles: 64, maxVertices: 64, ...options } );
	const box = boxOf( positions );
	set.boundingBox = { min: { x: box[ 0 ], y: box[ 1 ], z: box[ 2 ] }, max: { x: box[ 3 ], y: box[ 4 ], z: box[ 5 ] } };
	return set;

}

export function boxOf( positions ) {

	const b = [ Infinity, Infinity, Infinity, - Infinity, - Infinity, - Infinity ];
	for ( let i = 0; i < positions.length; i += 3 ) for ( let a = 0; a < 3; a ++ ) { b[ a ] = Math.min( b[ a ], positions[ i + a ] ); b[ 3 + a ] = Math.max( b[ 3 + a ], positions[ i + a ] ); }
	return b;

}

/** Column-major TRS matrix. */
export function compose( tx, ty, tz, yaw = 0, sx = 1, sy = sx, sz = sx ) {

	const c = Math.cos( yaw ), s = Math.sin( yaw );
	return new Float32Array( [ c * sx, 0, - s * sx, 0, 0, sy, 0, 0, s * sz, 0, c * sz, 0, tx, ty, tz, 1 ] );

}

/** A camera world matrix looking from `eye` at `target` (three.js convention: the camera looks down its local -Z). */
export function lookAt( eye, target, up = [ 0, 1, 0 ] ) {

	let zx = eye[ 0 ] - target[ 0 ], zy = eye[ 1 ] - target[ 1 ], zz = eye[ 2 ] - target[ 2 ];
	let l = Math.hypot( zx, zy, zz ); zx /= l; zy /= l; zz /= l;
	let xx = up[ 1 ] * zz - up[ 2 ] * zy, xy = up[ 2 ] * zx - up[ 0 ] * zz, xz = up[ 0 ] * zy - up[ 1 ] * zx;
	l = Math.hypot( xx, xy, xz ); xx /= l; xy /= l; xz /= l;
	const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
	return new Float32Array( [ xx, xy, xz, 0, yx, yy, yz, 0, zx, zy, zz, 0, eye[ 0 ], eye[ 1 ], eye[ 2 ], 1 ] );

}
