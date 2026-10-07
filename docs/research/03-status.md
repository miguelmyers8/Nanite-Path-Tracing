# Status and roadmap

## Done

| Stage | What it does | Where |
|---|---|---|
| 1 Acceleration structures | binned-SAH builder; a triangle BVH per cluster (leaves of four, an order byte per triangle); a hierarchy per mesh over every LOD level (one sub-tree per level joined at the top) with per-node largest parent error, smallest own error, the box of the LOD spheres and the largest parent radius; a per-frame CPU TLAS over instances with world and inverse matrices and scales; one packed vec4 buffer the kernel and the CPU twin both read | `src/core/` |
| 2 The tracer | a trace kernel (camera ray at bounce 0, TLAS → hierarchy with the two prunes and the cut rule → cluster BVH → triangles through the mesh's corner fetch) and a shade kernel (material table or flat colour, GGX + Lambert, a shadow ray towards the sun disk, a sampled next direction with one-sample MIS over the lobes, Russian roulette) per bounce; the path state in the frame buffer between them; progressive accumulation; a display quad with the debug views drawn from the primary hit record; a split against the raster | `src/three/PathTracePass.js` |
| 3 Validation | `node --test`: builders, layouts, hierarchy traversal equal to the brute force over the cut at many distances and thresholds, scenes with non-uniform scale, any-hit, forced levels, closed surfaces without cracks; headless WebGPU: the GPU's primary hits equal the CPU twin on every pixel, bounced frames finite, every view draws, the debugger's own verification | `test/`, `scripts/gpu-harness.mjs` |

## Open items

- **Paged and chunked meshes.** The kernel reads triangles through the mesh's
  corner fetch, which already decodes the compressed page pool and the
  camera-relative chunks, but `buildAccel` reads the uncompressed set and the
  hierarchy is static: a streamed set needs its cluster BVHs built when a
  page lands (the per-cluster trees could travel in the page), a residency
  term in the cut rule (the cull kernel's "resident and (fine enough or the
  finer page is absent)"), and the chunk origin added to the boxes; a pool
  set needs its hierarchy rebuilt when chunks install (cheap: it is per mesh
  and the pool packs chunks low).
- **Performance.** One sample per pixel per frame at one bounce is the
  real-time budget on a discrete GPU; convergence is progressive while the
  camera rests. Next: a ray-cone footprint for texture levels, compaction of
  live paths between the kernels (the split is where it goes), a half-float
  path state, four-wide hierarchy nodes, and reusing the previous frame
  (temporal accumulation with reprojection) while moving. The cost view and
  the per-ray cost in the hit record are the instrumentation.
- **Light transport.** The sun is sampled by next-event estimation, the sky
  only through the sampled bounce: fine for a smooth gradient, not for an
  environment map, which wants importance sampling with MIS. No emissive
  surfaces, no transmission, no area lights other than the sun disk.
- **The cut for secondary rays** is the camera's. That is what keeps every
  ray on one surface and what Unreal's streamed-out mode does with a global
  error; a coarser cut far from the camera for bounced rays (a second
  threshold) would save traversal where the camera cannot tell.
- **Denoising and upscaling.** None. three-gpu-pathtracer's OIDN and FSR
  hooks are the model.
- **An artifact build.** The Nanite repository packages its debuggers as
  self-contained folders for claude.ai Artifacts; this debugger loads the
  pipeline from `node_modules` and would need the same vendoring.

## What a production tracer has that this one does not

Hardware ray tracing (not in WebGPU), cluster acceleration structures built
by the driver, per-instance BLAS sharing and caching for instanced scenes,
streaming requests from the traversal, material binning for shading, a
denoiser, and a proxy representation for the far field and for global
illumination rays.
