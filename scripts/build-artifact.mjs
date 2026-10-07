/**
 * Package the path trace debugger as a self-contained folder that can be published as a claude.ai Artifact (or hosted on any
 * static server). The packaging follows the Nanite repository's own `scripts/build-artifact.mjs`; what differs is that the
 * pipeline is a package here (`node_modules/nanite`), so its source is copied next to this repository's.
 *
 *   node scripts/build-artifact.mjs [example]   → dist/<example>/            (default: pathtrace-debugger)
 *
 * Output layout:
 *   index.html     the page content only (title, links, style, body): the Artifact host wraps it in its own document skeleton
 *   preview.html   the same content inside a skeleton like the host's, for local tests (not published)
 *   app.js         the example
 *   src/           this repository's src/        (`nanite-path-tracing/...` imports)
 *   nanite/        the Nanite pipeline's src/    (`nanite/...` imports)
 *   vendor/        the three.js addons the example imports, their own imports rewritten
 *   ../<example>.files.json   the file list to publish
 *
 * Rewrites, because a published page has no import map (the host's own skeleton does not carry one):
 *  - `three`, `three/webgpu` → the pinned jsDelivr build of three.webgpu.js
 *  - `three/tsl` → destructured from the `TSL` export of that same build (three.tsl.js itself imports the bare 'three/webgpu')
 *  - `three/addons/...` → vendored under vendor/
 *  - `nanite/...`, `nanite-path-tracing/...` → relative paths into the copied trees
 * and the build fails when any bare specifier is left in any output module.
 */
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve( path.dirname( fileURLToPath( import.meta.url ) ), '..' );
const example = process.argv[ 2 ] || 'pathtrace-debugger';
const outDir = path.join( root, 'dist', example );
const srcExample = path.join( root, 'examples', example );

const threeDir = await fs.realpath( path.join( root, 'node_modules/three' ) );
const naniteDir = await fs.realpath( path.join( root, 'node_modules/nanite' ) );
const THREE_VERSION = JSON.parse( await fs.readFile( path.join( threeDir, 'package.json' ), 'utf8' ) ).version;
const CDN = `https://cdn.jsdelivr.net/npm/three@${ THREE_VERSION }`;
const CDN_WEBGPU = `${ CDN }/build/three.webgpu.js`;
const addonsDir = path.join( threeDir, 'examples/jsm' );
const naniteSrc = path.join( naniteDir, 'src' );
const naniteCommit = ( await fs.readFile( path.join( root, 'package.json' ), 'utf8' ).then( ( t ) => JSON.parse( t ).dependencies?.nanite ?? '' ).catch( () => '' ) ).split( '#' )[ 1 ] || 'local';

await fs.rm( outDir, { recursive: true, force: true } );
await fs.mkdir( outDir, { recursive: true } );

const vendored = new Set();
const rel = ( from, target ) => { let r = path.relative( from, target ).split( path.sep ).join( '/' ); if ( ! r.startsWith( '.' ) ) r = './' + r; return r; };

/** Rewrite the import specifiers of a module for the artifact layout. `fromDir`: where the file lands in the output. */
async function rewriteModule( code, fromDir ) {

	code = code.replace( /import\s*\{([^}]*)\}\s*from\s*(['"])three\/tsl\2\s*;?/g, ( _, names ) => {

		const list = names.split( ',' ).map( ( n ) => n.trim() ).filter( Boolean ).map( ( n ) => n.replace( /\s+as\s+/, ': ' ) );
		return `import { TSL as __TSL } from '${ CDN_WEBGPU }';\nconst { ${ list.join( ', ' ) } } = __TSL;`;

	} );
	code = code.replace( /import\s*\*\s*as\s+(\w+)\s*from\s*(['"])three\/tsl\2\s*;?/g, ( _, name ) => `import { TSL as ${ name } } from '${ CDN_WEBGPU }';` );

	const re = /(from\s*|import\s*\(\s*|import\s+)(['"])([^'"]+)\2/g;
	for ( const m of [ ...code.matchAll( re ) ] ) {

		const spec = m[ 3 ];
		let replacement = null;
		if ( spec === 'three' || spec === 'three/webgpu' ) replacement = CDN_WEBGPU;
		else if ( spec.startsWith( 'nanite/' ) ) replacement = rel( fromDir, path.join( outDir, 'nanite', spec.slice( 'nanite/'.length ) ) );
		else if ( spec.startsWith( 'nanite-path-tracing/' ) ) replacement = rel( fromDir, path.join( outDir, 'src', spec.slice( 'nanite-path-tracing/'.length ) ) );
		else if ( spec.startsWith( 'three/addons/' ) ) {

			const addonRel = spec.slice( 'three/addons/'.length );
			await vendorAddon( addonRel );
			replacement = rel( fromDir, path.join( outDir, 'vendor', addonRel ) );

		}

		if ( replacement ) code = code.replace( m[ 0 ], `${ m[ 1 ] }${ m[ 2 ] }${ replacement }${ m[ 2 ] }` );

	}

	return code;

}

async function vendorAddon( addonRel ) {

	if ( vendored.has( addonRel ) ) return;
	vendored.add( addonRel );
	const target = path.join( outDir, 'vendor', addonRel );
	await fs.mkdir( path.dirname( target ), { recursive: true } );
	let code = await fs.readFile( path.join( addonsDir, addonRel ), 'utf8' );
	// relative imports inside the addon: vendor those too
	for ( const m of [ ...code.matchAll( /(from\s*|import\s*\(\s*)(['"])(\.{1,2}\/[^'"]+)\2/g ) ] ) await vendorAddon( path.normalize( path.join( path.dirname( addonRel ), m[ 3 ] ) ).split( path.sep ).join( '/' ) );
	await fs.writeFile( target, await rewriteModule( code, path.dirname( target ) ) );

}

async function copyTree( from, to ) {

	await fs.mkdir( to, { recursive: true } );
	for ( const entry of await fs.readdir( from, { withFileTypes: true } ) ) {

		const s = path.join( from, entry.name ), d = path.join( to, entry.name );
		if ( entry.isDirectory() ) { if ( entry.name !== 'node_modules' ) await copyTree( s, d ); }
		else if ( entry.name.endsWith( '.js' ) ) await fs.writeFile( d, await rewriteModule( await fs.readFile( s, 'utf8' ), to ) );
		else await fs.copyFile( s, d );

	}

}

await copyTree( path.join( root, 'src' ), path.join( outDir, 'src' ) );
await copyTree( naniteSrc, path.join( outDir, 'nanite' ) );
for ( const entry of await fs.readdir( srcExample, { withFileTypes: true } ) ) {

	if ( entry.name === 'index.html' ) continue;
	const s = path.join( srcExample, entry.name ), d = path.join( outDir, entry.name );
	if ( entry.isDirectory() ) await copyTree( s, d );
	else if ( entry.name.endsWith( '.js' ) ) await fs.writeFile( d, await rewriteModule( await fs.readFile( s, 'utf8' ), outDir ) );
	else await fs.copyFile( s, d );

}

// --- the page: head content (title, links, style) then the body, without the document tags and the import map ----------------
const source = await fs.readFile( path.join( srcExample, 'index.html' ), 'utf8' );
const head = source.match( /<head>([\s\S]*?)<\/head>/i )?.[ 1 ] ?? '';
const body = source.match( /<body>([\s\S]*?)<\/body>/i )?.[ 1 ] ?? '';
const headKept = head.replace( /<meta[^>]*>\s*/gi, '' ).replace( /<script type="importmap">[\s\S]*?<\/script>\s*/i, '' ).trim();
const page = `${ headKept }\n${ body.trim() }\n`;
await fs.writeFile( path.join( outDir, 'index.html' ), page );

// a stand-in for the host's skeleton (doctype, charset, viewport-fit=cover, the light reset with the safe-area padding), so a local
// run is in standards mode like the published page; the host's own reset differs in detail
await fs.writeFile( path.join( outDir, 'preview.html' ), `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>:root{color-scheme:light;padding:env(safe-area-inset-top,0px) 0 env(safe-area-inset-bottom,0px)}body{margin:0;font:14px system-ui,sans-serif;background:#fafafa;color:#111}img{max-width:100%}[hidden]{display:none!important}</style>
${ headKept }
</head><body>
${ body.trim() }
</body></html>
` );

// --- checks: no bare specifier left, no reference outside the folder, every relative import resolves ------------------------------
const files = [];
async function walk( dir ) {

	for ( const entry of await fs.readdir( dir, { withFileTypes: true } ) ) {

		const p = path.join( dir, entry.name );
		if ( entry.isDirectory() ) await walk( p ); else files.push( p );

	}

}

await walk( outDir );
const problems = [];
const cdnAllowed = /^https:\/\/cdn\.jsdelivr\.net\/npm\/three@\d+\.\d+\.\d+\/build\/three\.webgpu\.js$/;
for ( const f of files.filter( ( f ) => f.endsWith( '.js' ) ) ) {

	// the lines that are code: doc comments mention "from 'disk'" and the like
	const code = ( await fs.readFile( f, 'utf8' ) ).split( '\n' ).filter( ( l ) => ! /^\s*(\*|\/\/|\/\*)/.test( l ) ).join( '\n' );
	for ( const m of code.matchAll( /(?:from\s*|import\s*\(\s*|import\s+)(['"])([^'"]+)\1/g ) ) {

		const spec = m[ 2 ];
		if ( spec.startsWith( 'https://' ) ) { if ( ! cdnAllowed.test( spec ) ) problems.push( `${ path.relative( outDir, f ) }: unexpected URL ${ spec }` ); continue; }
		if ( ! spec.startsWith( '.' ) ) { problems.push( `${ path.relative( outDir, f ) }: bare specifier '${ spec }'` ); continue; }
		const target = path.resolve( path.dirname( f ), spec );
		if ( ! target.startsWith( outDir + path.sep ) ) problems.push( `${ path.relative( outDir, f ) }: '${ spec }' leaves the folder` );
		else if ( ! files.includes( target ) ) problems.push( `${ path.relative( outDir, f ) }: '${ spec }' does not exist in the output` );

	}

}

if ( problems.length ) { console.error( problems.join( '\n' ) ); process.exit( 1 ); }

const publish = files.map( ( f ) => path.relative( outDir, f ).split( path.sep ).join( '/' ) ).filter( ( f ) => f !== 'preview.html' && f !== 'index.html' ).sort();
await fs.writeFile( path.join( root, 'dist', `${ example }.files.json` ), JSON.stringify( { page: 'index.html', files: publish, three: THREE_VERSION, nanite: naniteCommit }, null, 1 ) );
let bytes = 0; for ( const f of files ) bytes += ( await fs.stat( f ) ).size;
console.log( `built dist/${ example }: ${ files.length } files, ${ ( bytes / 1024 ).toFixed( 0 ) } KB, three ${ THREE_VERSION } from jsDelivr, nanite ${ naniteCommit.slice( 0, 8 ) }, vendored addons: ${ [ ...vendored ].join( ', ' ) || 'none' }` );
