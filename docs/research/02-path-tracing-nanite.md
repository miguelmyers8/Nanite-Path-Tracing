# Research notes: path tracing Nanite geometry

How the industry traces clustered, LOD-DAG, streamed geometry; what a WebGPU
without hardware ray tracing can do with it; and the design this project
chose. Compiled on 2026-10-07; several primary hosts (Epic's docs and forums,
docs.vulkan.org, the EG/ACM/Wiley libraries, arXiv) were unreachable from
the build sandbox, so some facts come from engine-source mirrors, sample
READMEs and search snippets and are marked where uncertain.

## Unreal Engine 5: three generations of Nanite ray tracing

**5.0, the fallback mesh.** A static simplified proxy per asset (a per-asset
"Fallback Relative Error" / triangle percent; in 5.6+ the "Ray Tracing
Proxy" settings) with an ordinary BLAS. Cheap; silhouettes and self-shadowing
disagree with the rasterized Nanite surface.

**5.1 to 5.6, the streamed-out mesh (`r.RayTracing.Nanite.Mode 1`).**
"Initial support for native ray-tracing of Nanite meshes… preserves all
detail while using significantly less GPU memory than zero-error Fallback
Meshes." From the engine source (5.3.2 and 5.6.0 mirrors):
`r.RayTracing.Nanite.CutError` (default 0) is a "global target cut error to
control quality", `MaxBuiltPrimitivesPerFrame` (8,388,608) and a staging
buffer size budget the rebuilds. `NaniteStreamOut.usf` runs a persistent
threads traversal of the same hierarchy the rasterizer culls with, but
against a constant object-space error, not a projected one:

    bShouldVisitChild  = StreamOutCutError < HierarchyNodeSlice.MaxParentLODError;   // node
    bSmallEnoughToDraw = StreamOutCutError > Cluster.LODError;                       // cluster
    bVisible = bSmallEnoughToDraw || (Cluster.Flags & NANITE_CLUSTER_FLAG_STREAMING_LEAF);

So the cut is view independent ("own error < CutError < parent error"), a
cluster whose finer pages are not resident counts as a leaf, the selected
clusters are decoded into staging vertex and index buffers, and ordinary
BLAS builds follow, throttled by the budgets. The ray-traced geometry is
whatever is resident at `CutError`, never the rasterized cut. NVIDIA's 5.4
guideline notes the mode "will generally perform slower than the Nanite
Fallback mesh due to the higher cost of building raytracing acceleration
structure for denser geometry". An older experiment, `NaniteRayTrace.ush`
behind `r.RayTracing.Nanite.ProceduralPrimitive`, evaluated `CutError >
Cluster.LODError` inside the ray tracing shaders: the LOD test in the
traversal.

**5.6+ with RTX Mega Geometry.** Stock 5.6.0 lists modes 0 and 1; the
cluster path shipped in NVIDIA's NvRTX branch (Karis: "RTX Mega Geometry
enables ray tracing with extreme detail and geometric complexity in a way
that wasn't possible before"), and Gears of War: E-Day (UE 5.8 + NvRTX,
2026-10) is the first shipping game. Third-party 5.8 projects describe a
`r.RayTracing.Nanite.Mode 2` that hands cluster vertex and index GPU
addresses to NVAPI cluster builds with no stream-out decompression; whether
that is mainline or NvRTX-only could not be confirmed.

## NVIDIA RTX Mega Geometry (2025 onward)

`VK_NV_cluster_acceleration_structure` (and the same through NVAPI on
D3D12): a **CLAS** is an acceleration structure over one cluster (up to 256
triangles, with a 32-bit `ClusterID` readable in hit shaders and optional
quantization by zeroing mantissa bits); a **Cluster BLAS** is built from
CLAS *addresses*, "not copied in like in a traditional triangle BLAS, which
allows more memory control and re-use across BLAS in cluster-based level of
detail schemes"; cluster templates precompute what is independent of the
final vertex positions; `VK_NV_partitioned_tlas` updates part of a TLAS.
The motivation named in the proposal: "managing numerous animated objects,
implementing LOD systems, or handling dynamic tessellation".

The per-frame flow of `nvpro-samples/vk_lod_clusters`: `traversal_init`
seeds the instance LOD roots, `traversal_run` (a persistent kernel) walks
the LOD hierarchy and outputs the render cluster list, `blas_setup_insertion`
and `blas_clusters_insert` fill per-BLAS cluster reference ranges, indirect
Cluster-BLAS builds, a TLAS build, then the trace. CLAS are built once when
a group streams in; "traversal of LoD hierarchy and interaction with
streaming system are the same" for raster and ray tracing: one cut feeds
both. Instanced scenes build many BLAS per frame, hence BLAS sharing (a
conservative per-geometry LOD from the instance sphere's nearest and
farthest points), caching across frames, and merging. The LOD builder
(`nv_cluster_lod_builder`) is Nanite's: cluster, group, decimate to half
with locked group borders, a quadric error per group that "must never
decrease with each level", and bounding spheres that "conservatively include
their generating group's bounding spheres. This guarantees that clusters
from multiple levels cannot be rendered at once." Its runtime rule, select a
cluster when `errorOverDistance(generating group) < threshold` and
`errorOverDistance(group) >= threshold`, is this pipeline's cull rule.
RTXMG 2.0 (2026-08) adds a Cluster LOD path with GPU-driven residency
against fixed pools, occlusion feeding back into LOD selection, and the
Zorah scene at 56 M unique / 778 M instanced triangles in 15.5 ms at 4K.

Transferable to software: the DAG data, monotonic errors with nested
spheres, one cut shared by raster and rays, conservative per-geometry LOD for
instances, cross-frame caching. Hardware bound: CLAS and Cluster-BLAS
objects and their driver builds, templates, hit-shader cluster ids,
partitioned TLAS.

## Research

- Benthin and Peters, "Real-Time Ray Tracing of Micro-Poly Geometry with
  Hierarchical Level of Detail" (HPG 2023, CGF 42(8)): cluster, merge and
  simplify into an HLOD DAG, compress into a GPU-friendly format; "at
  runtime, each selected cluster is decompressed into a small BVH in the
  format expected by ray tracing hardware, and then a complete BVH is built
  on top of these cluster bounding volumes": per-frame selection, per-cluster
  BVHs from a lossy cache, a per-frame top-level rebuild. The RTX Mega
  Geometry pattern, two years earlier.
- Intel's traversal shaders ("Flexible Ray Traversal with an Extended
  Programming Model", SA 2019: programmable instance selection, stochastic
  LOD) and "Lazy Build of Acceleration Structures with Traversal Shaders"
  (SA 2020); Lloyd et al. 2020 (NVIDIA), stochastic LOD by cross-dissolving
  two levels per pixel; Haydel, Yuksel, Seiler, "Locally-Adaptive
  Level-of-Detail for Hardware-Accelerated Ray Tracing" (SA 2023), the level
  chosen during traversal.
- 2025 to 2026: HPG 2026 "Ray Tracing Massive Amounts of Animated Geometry"
  (AMD: two-level structures over tetrahedral cages), "Memory-Efficient BVHs
  with Merged Nodes", Ladeuil et al. "Construction of clustered HLOD with
  As-Simplified-As-Possible boundaries" (CGF 2026). Practitioner work:
  jglrxavpok's "Recreating Nanite: Raytracing" (2024) and MetalRenderer PR
  #38 (the raster cluster list reused as the tracer's selection).

## WebGPU

- **No hardware ray tracing.** gpuweb/gpuweb#535 "Ray Tracing extension"
  (opened 2020, milestone "4+") is open with no working-group proposal as of
  2026-06; wgpu's ray queries are native-only.
- **Software tracers** exist and are fast enough: three-gpu-pathtracer's
  WebGPU backend (wavefront and megakernel compute, indirect dispatch,
  ray-queue ring buffers, 30-entry traversal stacks, one in workgroup
  memory); three-mesh-bvh's `bvh_ray_functions.wgsl.js` traverses with
  `var stack: array<u32, N>`; its `ClusteredBVH` is a meta-BVH over
  sub-trees of source BVHs (7-bit root + 24-bit node).
- **TSL**: `array( type, n ).toVar()` emits a function-scope (in compute, a
  private) `array<T, n>` with dynamic indexing, `Loop` with `Break`, so short
  stacks work; three-mesh-bvh deprecated passing storage buffers into WGSL
  functions (a WGSL restriction), so a traversal is inlined where it is used.
  Sorting is not built in (three ships a bitonic sort example; WGSL radix
  sorts exist), which weighs against per-frame LBVH builds.
- **Scthe/nanite-webgpu** has the meshlet DAG, a software rasterizer and
  culling, no ray tracing.

## The architectures for this pipeline

Given per cluster a bounding sphere, a normal cone, own and parent errors
(monotonic, nested spheres) and a cull kernel that selects the cut with
"own projected error ≤ threshold < parent projected error":

**(a) A per-frame top level over the cut plus prebuilt per-cluster BVHs**
(Intel 2023, Mega Geometry). Per frame: cull → compact cluster list → a top
level over 10⁴ to 10⁵ cluster boxes (LBVH: Morton codes, a radix sort,
Karras; three to five dispatches and a refit, with a sort to bring; or a CPU
build from a readback, one frame late). Smallest live tree and the fastest
rays, exactly the raster cut if fed the same list. But a build every frame,
and the raster cut is frustum and occlusion culled, so what shadow and GI
rays need off screen is missing unless a second, uncilled cut is selected.

**(b) A static hierarchy over all levels, pruned by the error metric during
traversal** (the Nanite culling hierarchy used as the BVH, UE's
`NaniteRayTrace.ush` experiment, Intel's traversal shaders). Internal nodes
carry the subtree's largest parent error and smallest own error with the
box the distances are measured against; a ray descends a node only if the
largest parent error projects above the threshold at the nearest possible
distance (otherwise an ancestor's level already covers it) and the smallest
own error projects below it at the farthest (otherwise everything below is
too coarse); at a cluster it applies the cull kernel's own test, with the
error projected from the camera for every ray, so all rays see one
crack-free surface identical to the raster cut. No per-frame build, no sort,
nothing missing off screen. The tree holds every level, so it is about
twice the cut's size, and each visited node costs a box distance and a
divide.

**(c) A proxy mesh** at a fixed error: trivial, wrong silhouettes; sensible
as a far-field or GI representation later.

## Decision, and what was built

(b), with (a)'s cluster-level piece: per-cluster triangle BVHs are built
once (a software CLAS), the per-mesh hierarchy is built once with one
sub-tree per LOD level joined at the top (so the prunes remove whole levels
at once), and the TLAS over instances is a small CPU BVH rebuilt per frame
from the matrices the CPU already owns. Everything is packed into vec4
buffers the CPU reference reads too (`src/core/`), so the GPU kernel and
the JavaScript twin traverse the same bytes.

The tracer is two TSL compute kernels per bounce, wavefront style (a trace
kernel for the path ray, a shade kernel with the sun's shadow ray and the
next direction, the path state in a storage buffer between them), and a
display quad. Triangles are read through the mesh's own corner fetch, the
decode every draw path uses, so a paged or chunked mesh could trace with the
same kernel once its acceleration data exists. Shading keeps the raster's
light convention (a Lambert surface facing the sun returns `albedo ×
lightColor`; the hemisphere sky and ground colours are the environment) and
samples the material table at the hit.

Measured in `test/gpu/pathtrace-parity.html` (headless Chrome, SwiftShader,
four instances of two meshes, 401 clusters, 96 × 72): the GPU's primary
hits (instance, cluster, triangle) equal the CPU reference on all 6,912
pixels; the average primary ray tests 37 nodes and triangles; a 320 × 240
frame with three bounces takes 1.5 s on SwiftShader (a CPU), the number to
divide by a real GPU's throughput.

## Lessons from the implementation

- The two prunes must be conservative under non-uniform scale: the nearest
  world distance is at least `minScale × distance(cameraObject, box) −
  maxParentRadius × maxScale`, the farthest at most `maxScale × farDistance`;
  the leaf test is then exact, in world space, with the same expressions as
  the cull kernel, and the CPU twin calls Nanite's `lodSelected` itself.
- Every level of a torus has the same bounding box, so a top-level builder
  over level roots can put several levels in one leaf; the hierarchy
  emitter handles multi-item leaves with a chain of internal nodes, and the
  generic builder splits coincident centroids at the median instead of
  making a big leaf.
- Headless validation needs care: three r185 passes the texture-view
  `swizzle` field as a string that Playwright's Chromium 141 validates as a
  dictionary (the harness strips it), and SwiftShader cannot create a canvas
  swap chain, so every test renders into a render target. A device lost on
  present looks exactly like a crashed kernel; the GPU process log
  (`--enable-logging=stderr`) tells them apart.
- `getArrayBufferAsync( attribute, null, byteOffset, byteLength )` reads
  one region of a storage buffer: the per-pixel hit records are 16 bytes a
  pixel, read back for the parity check and the cost view without the
  colour sums.

## Sources

- Epic, Nanite Virtualized Geometry and Hardware Ray Tracing docs: <https://dev.epicgames.com/documentation/unreal-engine/nanite-virtualized-geometry-in-unreal-engine>, <https://dev.epicgames.com/documentation/unreal-engine/hardware-ray-tracing-in-unreal-engine>
- UE source mirrors (5.3.2 `NaniteStreamOut.usf`, `NaniteRayTracing.cpp`; 5.6.0 cvars): <https://github.com/chenyong2github/UnrealEngine>, <https://github.com/lixiang518/UE5.6.0_D3D12RHI>
- NVIDIA UE5 Raytracing Guideline v5.4: <https://dlss.download.nvidia.com/uebinarypackages/Documentation/UE5+Raytracing+Guideline+v5.4.pdf>; Gears of War: E-Day: <https://www.nvidia.com/en-us/geforce/news/gears-of-war-e-day-dlss-4-5-ray-tracing-rtx-mega-geometry/>
- Karis, Nanite deep dive (SIGGRAPH 2021): <https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf>
- `VK_NV_cluster_acceleration_structure` proposal: <https://github.com/KhronosGroup/Vulkan-Docs/blob/main/proposals/VK_NV_cluster_acceleration_structure.adoc>; NVIDIA blog: <https://developer.nvidia.com/blog/nvidia-rtx-mega-geometry-now-available-with-new-vulkan-samples>
- vk_lod_clusters, nv_cluster_lod_builder, vk_animated_clusters, RTXMG: <https://github.com/nvpro-samples/vk_lod_clusters>, <https://github.com/nvpro-samples/nv_cluster_lod_builder>, <https://github.com/nvpro-samples/vk_animated_clusters>, <https://github.com/NVIDIA-RTX/RTXMG>
- Benthin and Peters, HPG 2023: <https://onlinelibrary.wiley.com/doi/10.1111/cgf.14868>
- Intel traversal shaders: <https://www.intel.com/content/dam/develop/external/us/en/documents/flexible-ray-traversal-with-an-extended-programming-model-839978.pdf>; lazy build: <https://www.intel.com/content/www/us/en/developer/articles/technical/lazy-build-of-acceleration-structures.html>; Haydel, Yuksel, Seiler: <https://dl.acm.org/doi/10.1145/3618359>
- HPG 2026: <https://dl.acm.org/doi/full/10.1145/3820014>, <https://dl.acm.org/doi/10.1145/3820018>; clustered HLOD: <https://diglib.eg.org/items/db6dc9ab-a03d-4092-bcb2-4f200d5058d8>
- jglrxavpok, Recreating Nanite: Raytracing: <https://jglrxavpok.github.io/2024/08/21/recreating-nanite-raytracing.html>; MetalRenderer PR #38: <https://github.com/IIMrFreemanII/MetalRenderer/pull/38>
- gpuweb #535: <https://github.com/gpuweb/gpuweb/issues/535>; three-gpu-pathtracer PRs #713, #770; three-mesh-bvh `src/webgpu`: <https://github.com/gkjohnson/three-mesh-bvh/tree/master/src/webgpu>; nanite-webgpu: <https://github.com/Scthe/nanite-webgpu>
- Stackless traversal: Hapala et al. 2011 <https://dcgi.fel.cvut.cz/~havran/ARTICLES/sccg2011.pdf>, Binder and Keller 2016, Ylitie et al. 2017 <https://research.nvidia.com/sites/default/files/publications/ylitie2017hpg-paper.pdf>
