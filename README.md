# tokgan_after_effects_import

Import Tokgan ML shape data directly into Adobe After Effects as native shape
layers, bypassing the Silhouette → Adobe Premiere/AE export round-trip used
by the sibling [tokgan_silhouette_import][sib] tool.

For a **v2** JSON (or with `--flat`), the output is **one** AE shape layer
named *Tokgan Shapes* containing one Vector Group per source object. Each
group has:

* a Bezier path animated frame-by-frame,
* a hold-stepped opacity track driven by which source frames the object
  exists in,
* a fill.

Because every group is a true shape-layer path (not a mask), AE's regular
**motion blur switch** produces sub-frame interpolated blur — which was the
original motivation. The layer is created with motion blur already enabled.

[sib]: https://github.com/samhodge-aiml/tokgan_silhouette_import

For a **v3** JSON (Rotobot Next 0.10.0 and later, which adds the plate
camera and per-person data) it builds a camera / person / body-part
**hierarchy** instead — see below.

## Schema compatibility

Accepts `lozenge_bezier_anim` schema **v2 and v3**. v2 JSONs import as the
single flat layer, unchanged. v3 JSONs build the hierarchy; `--flat` gives
the single layer for them too.

## The v3 hierarchy

```
main comp
  Tokgan Shapes          precomp layer + Corner Pin (keyed every frame)
                         = the plate camera
    Tokgan Stabilised    precomp: the camera removed
      p0_pelvis          Null per person, position keyed every frame
        p0:arm:L:forearm shape layer per body part, parented to the person:
        p0:leg:R:thigh   position + rotation from the bone, path in
        ...              bone-local pixels
```

* **Camera — exact.** Rotobot's camera solve is a full perspective
  homography. AE can't parent a perspective transform, so the camera is a
  **Corner Pin** on the precomp layer: four points that reproduce the
  homography exactly (an affine approximation was up to 236 px off at the
  plate corners on a real 4K clip). Turn the Corner Pin off to see the
  stabilised rig.
* **Person.** One Null per person at the pelvis; move it to move the whole
  person.
* **Body part.** One shape layer per part, positioned and rotated by its
  bone, so a limb can be adjusted without touching its vertices.
* The stabilised precomp is sized to fit everything once the camera is
  removed (a long pan makes it wider than the plate).
* Missing or failed tracking (a failed camera frame, no hips detected, a
  frame without a bone) holds the last good value instead of jumping; the
  converter reports how many frames were held.
* Before writing anything the converter recomposes every vertex through
  the hierarchy and refuses if any lands more than 0.05 px from the source.
  The decomposition is
  [`rotobot-nuke`](https://github.com/samhodge-tokgan/rotobot-nuke)'s
  `rotobot_nuke.hierarchy`, shared with the Nuke and Silhouette importers and
  vendored here as `_rotobot_hierarchy/`
  (`tools/check-vendored-hierarchy.sh` checks it against the pinned commit).

Position, rotation and Corner Pin keys are linear, so a JSON thinned with
`rotobot-undersample` (below) interpolates the way its error metric
assumed.

### Reducing keyframe count before import

The converter writes **every** frame of each spline as an AE keyframe.
On a 24 fps action clip that's ~ 24 keyframes per second per spline
per person — AE can scrub it, but hand-tweaks become tedious.

For an actor whose articulation is captured in a Rotobot-Next v3 JSON,
pre-pass through [`rotobot-undersample`](https://pypi.org/project/rotobot-nuke/)
before running this converter:

```bash
pip install rotobot-nuke        # provides the `rotobot-undersample` CLI
rotobot-undersample shapes.json shapes_reduced.json --preset balanced
python tokgan_json_to_ae.py shapes_reduced.json
```

The `balanced` preset keeps ~ 89% of keyframes on a measured 4-clip
real-plate benchmark (3 pp spread across clips), dropping the ones
that are linear interpolations of their neighbours in a composed
camera + person-root + body-local-articulation metric. See
[`rotobot-nuke/benchmarks/real_results_cross_clip.md`](https://github.com/samhodge-tokgan/rotobot-nuke/blob/main/benchmarks/real_results_cross_clip.md)
for the methodology + fine/coarse alternatives.

The pre-pass is optional — direct conversion still works unchanged.

## Requirements

* Python 3.9+, standard library only — nothing to install. The v3
  hierarchy code is vendored beside the script in `_rotobot_hierarchy/`
  (from [rotobot-nuke](https://github.com/samhodge-tokgan/rotobot-nuke), MIT);
  keep that folder next to `tokgan_json_to_ae.py`.
* After Effects 2024+ (tested on AE 25.6 / Mac).
* The AE preference **Allow Scripts to Write Files and Access Network**
  must be enabled — `Settings > Scripting & Expressions`. Without it the
  loader cannot read its sidecar JSON.

## Quick start

```bash
# Convert a JSON to an AE loader + sidecar
python3 tokgan_json_to_ae.py data/9961755_uhd_2160x4096_25fps_48frames.json

# This writes two files next to the JSON:
#   data/9961755_uhd_2160x4096_25fps_48frames.jsx
#   data/9961755_uhd_2160x4096_25fps_48frames_data.json
```

Then in After Effects:

1. Open (or create) a composition whose dimensions match the JSON's
   `resolution` field. The loader uses `app.project.activeItem` if it's a
   comp; otherwise it creates a new `TokganShapes` comp.
2. `File > Scripts > Run Script File…` and pick the generated `.jsx`.
3. Wait — large clips can take 20–60 seconds for the path-keyframe writes.
   A progress line goes to the ExtendScript Toolkit console (`$.writeln`)
   when complete:
   ```
   [Tokgan] data fps=24 comp fps=25 scale=0.96
   ```

To replace an existing import without manual cleanup, just re-run the
loader. It removes the previous *Tokgan Shapes* layer before building the
new one.

### Usage

```
$ python3 tokgan_json_to_ae.py --help
usage: tokgan_json_to_ae [-h] [-f] [--keep-background] [--fps FPS] [--flat]
                         input [output]

Convert a Tokgan JSON shape file into an After Effects loader (.jsx) plus a
compact sidecar data JSON. Run the resulting .jsx in AE via File > Scripts >
Run Script File... A v2 JSON builds one shape layer holding an animated Bezier
path per object; a v3 JSON (camera + person data) builds a camera Corner Pin,
a Null per person and a shape layer per body part.

positional arguments:
  input              Path to input Tokgan JSON file
  output             Path to write the AE loader .jsx (default: same dir/name
                     as the input with .jsx extension). The sidecar will be
                     written next to it as <output_stem>_data.json.

options:
  -h, --help         show this help message and exit
  -f, --force        Rebuild the sidecar data file even if it's newer than the
                     input.
  --keep-background  Import 'background' persons (max-over-time bbox dim <
                     max(comp_w, comp_h)/20) as a single neutral-grey fill
                     instead of dropping them. Off by default to save AE
                     keyframe-writing time on crowded clips.
  --fps FPS          Override the fps declared in the JSON metadata. Tokgan
                     exports stamp every file with fps=24 even when the source
                     video runs at e.g. 25 fps, which makes the shapes play
                     ~4% faster than the matching footage in AE. Pass the
                     footage's true fps (e.g. --fps 25) to fix the timing.
  --flat             Always build the single 'Tokgan Shapes' layer, even for a
                     v3 JSON with camera/person data (which otherwise builds
                     the camera / person / body-part hierarchy).

Notes: each source frame lands on one comp frame regardless of the JSON's
declared fps. Enable 'Allow Scripts to Write Files and Access Network' in AE's
preferences (Scripting & Expressions) before running the loader.
```

## What the loader does, in order

1. Locates its sidecar `_data.json` next to itself via `parent.getFiles()`
   (avoids the ExtendScript File-constructor quirks with `:` and spaces in
   filenames).
2. Reads + parses the sidecar (`JSON.parse`, with `eval("("+raw+")")`
   fallback for ExtendScript builds that lack `JSON`).
3. Uses the active comp, or creates one with the JSON's resolution / fps /
   duration if none is active.
4. Removes any pre-existing *Tokgan Shapes* layer for clean re-runs.
5. For each source object, adds a Vector Group whose Contents holds a Path
   and a Fill. The path bezier is written in one batched
   `setValuesAtTimes(...)` call, with a per-frame fallback. The group's
   `Transform > Opacity` gets hold-stepped 0/100 keys around every visible
   segment.
6. Resets the layer's anchor and position to `(0,0)` so JSON pixel coords
   land at comp pixel coords with no half-resolution offset.
7. Scales every keyframe time by `data.fps / comp.frameRate` so each source
   frame lands on one comp frame regardless of declared fps.
8. Enables `layer.motionBlur` and `comp.motionBlur`.

## Example

`data/9961755_uhd_2160x4096_25fps_48frames.json` is a 23 MB sample from the
`tokgan_silhouette_import` repo — 48 frames at 25 fps, 63 body-part shapes
on a single subject. Run the converter, drop the `.jsx` into AE, and you'll
get a 2160 × 4096 comp populated with motion-blur-ready animated shape
paths.

The source footage for this clip is Pexels video **9961755** (UHD 2160 ×
4096, 25 fps); download from <https://www.pexels.com> and import as an
image sequence or video to match against the imported shapes.

## Sharing with other machines

The `.jsx` is only a loader; the shape data lives in the `_data.json`
sidecar. **Always send both files together** (same folder, original
names). Sending just the `.jsx` gets you a "Locate sidecar" dialog in AE,
and picking the raw Tokgan JSON there will not work. Alternatively, send
only the raw Tokgan JSON and have the recipient run the converter.

Output is identical on Mac, Linux and Windows: the `.jsx` is pure ASCII
with LF line endings, and the sidecar is UTF-8 with LF line endings.

## Compositing

To use the imported shape layer as an animated alpha matte on footage:

1. Drop your footage into the same comp, below the *Tokgan Shapes* layer.
2. On the footage layer, set the **Track Matte** column to *Tokgan Shapes*
   (defaults to Alpha Matte).

With a v3 hierarchy, *Tokgan Shapes* is the precomp layer carrying the
camera Corner Pin; use it as the Track Matte in exactly the same way.

Motion blur on the shape layer carries through to the matte edges. In a
v3 hierarchy the body-part layers blur inside the stabilised precomp; the
camera move itself is a Corner Pin effect, which AE does not motion-blur. For
naturally-centered blur, set the comp's **Shutter Phase** to half the
negative **Shutter Angle** (e.g. `-90°` when shutter is `180°`) under
`Composition Settings > Advanced`.

## AE 25.6 quirks the loader works around

While building this, I tripped on a handful of After Effects 25.6
ExtendScript quirks worth knowing about if you hack on the loader:

* `app.beginSuppressDialogs()` takes **zero** parameters in 25.6 (older
  docs say a boolean). Passing one throws `requires 0 parameters`. The
  loader doesn't call it at all.
* `JSON` is **undefined** in this ExtendScript build. The loader falls
  back to `eval("(" + raw + ")")` for the sidecar.
* AE invalidates earlier `Property` references when **sibling**
  `addProperty` calls mutate the parent group. The loader does all
  `addProperty` calls first, then re-walks `numProperties` to acquire the
  pathItem fresh — otherwise `pathProp.setValueAtTime(…)` throws
  `Object is invalid`.
* AE caches compiled `.jsx` bytecode by path. If you edit a loader and
  AE keeps replaying the old error, quit + relaunch AE or save to a
  fresh filename.
* `addProperty` / `canAddProperty` will hard-crash AE on undocumented
  match names — stick to the documented `ADBE …` strings.

## File layout

```
tokgan_after_effects_import/
├── README.md
├── LICENSE                                       MIT
├── tokgan_json_to_ae.py                          The converter
└── data/
    └── 9961755_uhd_2160x4096_25fps_48frames.json Example input
```

After a run, two more files appear next to the input:

```
data/9961755_uhd_2160x4096_25fps_48frames.jsx        AE loader script
data/9961755_uhd_2160x4096_25fps_48frames_data.json  Sidecar (gitignored)
```

## License

MIT — see [LICENSE](LICENSE).
