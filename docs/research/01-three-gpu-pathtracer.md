# Research notes: three-gpu-pathtracer, WebGPURenderer, and build-on versus build-own

The first question of this project: can the path tracer be built on top of
`three-gpu-pathtracer`, and does that library run on three.js's
`WebGPURenderer`, which is what the Nanite pipeline is written for (r185,
TSL compute)? Checked on 2026-10-07 against the npm package, the repository
and its issue tracker.

## Does three-gpu-pathtracer work with WebGPURenderer?

**Yes, since v0.0.25 (npm, 2026-09-28), through a new `WebGPUPathTracer`;
the old `WebGLPathTracer` is WebGL-only and deprecated.**

- `package.json` 0.0.26 (`latest`): `peerDependencies: { three: ">=0.185.0",
  "three-mesh-bvh": ">=0.9.15" }`, `exports: { ".": "./src/index.js",
  "./webgpu": "./src/webgpu/index.js" }`.
- CHANGELOG 0.0.25: "Added: WebGPUPathTracer, available from
  `three-gpu-pathtracer/webgpu`. Changed: Minimum three.js version is now
  r185. Deprecated WebGLPathTracer. It will be removed in a future release."
  0.0.26 fixed a missing file extension that broke CDN loading of the
  WebGPU tracer.
- README usage: `import { WebGPUPathTracer } from 'three-gpu-pathtracer/webgpu';
  renderer = new THREE.WebGPURenderer(); await renderer.init(); pathTracer =
  new WebGPUPathTracer( renderer ); pathTracer.setScene( scene, camera );
  pathTracer.renderSample();` and the gotcha "The project requires WebGPU":
  the real WebGPU backend, not `WebGPURenderer`'s WebGL2 fallback.
- The source under `src/webgpu/` is TSL/WGSL: 41 files import `three/tsl`,
  46 contain WGSL, kernels dispatch with `renderer.compute( kernel,
  dispatchSize )` (`MegaKernelPathTracer.js`, `WaveFrontPathTracer.js`,
  `PathTracerBackend.js`). The legacy path (`src/materials/MaterialBase.js
  extends ShaderMaterial`, 50 GLSL files, `gl_FragCoord`) only runs on
  `WebGLRenderer`; `WebGLPathTracer.js` warns that it is deprecated and calls
  `renderer.getContextAttributes()`.
- Maintainer statements: PR #713 "WebGPU Support" (opened 2026-02-04, merged
  2026-09-28: wavefront + megakernel compute kernels, storage textures,
  ring-buffer work queues); issue #779 (2026-07-09) "WebGLPathTracer will be
  deprecated in favor of WebGPUPathTracer"; issue #692 (2025-09) the
  architecture plan (WGSL + TSL compute, materials as structures, a TLAS,
  compressed wide BVH); issue #777 (open, milestone v0.0.27) the remaining
  features (fog, subsurface, some glTF extensions, async BVH, ray sorting);
  issue #868 (2026-09-30, open) a kernel compile failure at 1×1 with r186.
- three.js itself ships `webgpu_renderer_pathtracer` (updated 2026-09-29),
  which wraps `WebGPUPathTracer`; three has no path tracer of its own.
- three-mesh-bvh 0.9.15 exports `three-mesh-bvh/webgpu` (`BVHComputeData`,
  `ClusteredBVH`, `bvh_ray_functions.wgsl.js`, TSL structs), "requires
  three.js r185 or higher", "API is unstable", and 0.9.11 deprecated its
  `wgslFn`-based exports "due to WGSL restrictions on passing storage buffer
  pointers across function call boundaries".

So the library is two weeks into its WebGPU life, usable, and moving.

## Can it trace the Nanite pipeline's geometry?

Not the cut. `WebGPUPathTracer.setScene()` builds a `MeshBVH` per geometry
on the CPU (three-mesh-bvh, SAH, synchronously), a `ClusteredBVH` top level
over the per-object trees, and packs everything into storage buffers
(`bvh_nodes`, `bvh_transforms`, `bvh_index`, `bvh_attributes`,
`bvh_materials`). Its documentation: "The scene is captured when `setScene`
is called… Changing geometry requires setScene instead, since the BVH must
be rebuilt." Only transforms, a per-object visibility flag, materials, lights,
the camera and the environment update incrementally (`updateTransforms()`
refits the TLAS).

The Nanite pipeline has no mesh in that sense. Its surface is a cut of a
cluster DAG chosen on the GPU every frame from the camera's distance to each
cluster ("own projected error ≤ threshold < parent projected error"), so
the set of triangles changes continuously as the camera moves and the CPU
never sees per-cluster data. Feeding it to `setScene()` means either:

1. **all leaf clusters as one static mesh**: the full-resolution geometry,
   a CPU BVH over every triangle, no LOD at all (UE 5.0's "fallback mesh"
   at zero error), which is exactly the memory and build cost Nanite exists
   to avoid; or
2. **one object per cluster of every level, toggled with the visibility
   flag** after a cut readback: thousands of objects in the clustered top
   level, a one-frame-late cut, a CPU in the loop; or
3. **a proxy at a fixed error** (UE 5.1's `r.RayTracing.Nanite.CutError`
   idea): a view-independent cut baked to a static mesh, rebuilt when the
   target error changes. Workable for a far-field or GI proxy; not the
   rasterized surface.

None traces the cut the rasterizer draws. The one component that makes it a
Nanite path tracer, the acceleration structure and its LOD-aware traversal,
is the one the library cannot provide, and its kernels are tied to
three-mesh-bvh's node layout (`bvh_ray_functions.wgsl.js`), so the
intersection stage is not pluggable without forking `src/webgpu/`.

## Decision

Build the traversal ourselves, on the Nanite data the pipeline already has,
and keep the rest small: a static cluster hierarchy per mesh over all LOD
levels that a ray prunes with the error metric while it walks it, prebuilt
per-cluster triangle BVHs, a per-frame CPU TLAS over instances, and a
wavefront-style pair of TSL compute kernels for tracing and shading
(`docs/research/02-path-tracing-nanite.md` has the design and its sources).
Shading reuses the pipeline's material table and lighting conventions, so
the traced image matches the raster's cut, materials and light.

What three-gpu-pathtracer remains useful for:

- a **reference renderer** for a static proxy of the same scene (option 3
  above), to compare noise, convergence and materials against;
- **techniques to borrow** as the tracer grows: its wavefront kernel split
  with ray queues and indirect dispatch (PR #713), its MIS and environment
  sampling, the OIDN denoiser and FSR upscaler hooks, `dynamicLowRes`
  while moving;
- if a future three-mesh-bvh exposes a GPU builder (PR #853 is open), the
  per-frame top-level alternative in the design notes.

## Sources

- three-gpu-pathtracer repository, README, CHANGELOG, `package.json`: <https://github.com/gkjohnson/three-gpu-pathtracer>
- PR #713 WebGPU Support: <https://github.com/gkjohnson/three-gpu-pathtracer/pull/713>; PR #770 (traversal stacks, workgroup memory): <https://github.com/gkjohnson/three-gpu-pathtracer/pull/770>
- Issues #692, #777, #779, #868: <https://github.com/gkjohnson/three-gpu-pathtracer/issues>
- three-mesh-bvh releases and `src/webgpu`: <https://github.com/gkjohnson/three-mesh-bvh/releases>, <https://github.com/gkjohnson/three-mesh-bvh/tree/master/src/webgpu>; GPU builder PR #853: <https://github.com/gkjohnson/three-mesh-bvh/pull/853>
- three.js example `webgpu_renderer_pathtracer`: <https://github.com/mrdoob/three.js/blob/dev/examples/webgpu_renderer_pathtracer.html>
