# CoherentSearch (C)

A C port of [`CoherentSearch.jl`](https://github.com/scottransom/CoherentSearch.jl),
the coherent harmonic-summing pulsar search that folds PRESTO-style `.fft` files
using the **phase** of every harmonic.  The numerical algorithm follows the Julia
reference implementation; the candidate output is byte-identical on every
configuration tested, while the hot kernels are faster in C.

It also accepts raw `.dat` time series: the program performs PRESTO's `realfft`
and `rednoise` steps **in memory** (nothing is written to disk) and then runs the
same search.  A second program, `coherent_search_list`, processes a
`.dat.list` DM sweep while sharing the FFT plans, workspaces, plan tables and
CUDA context across all files.  Candidate profile plots are produced as
dependency-free SVG pages with `--plot`, and the two Python tools from the Julia
`bin/` directory (`sift_candidates.py`, `parallel_search.py`) are included and
work without Julia.

## AI assistance

This C port, including the CUDA backend, build files, and the Julia-independent
Python tools, was rewritten with assistance from **DeepSeek**.

## References

- Original Julia implementation:
  <https://github.com/scottransom/CoherentSearch.jl>
- Paper: <https://arxiv.org/abs/2610.24287>

## Build

```sh
source CoherentSearch_C/bashrc
cd CoherentSearch_C/CoherentSearch_C
make            # CPU-only binary (GPU=0, the default)
make GPU=1      # also build the CUDA/cuFFT backend
```

This builds two programs:

```
bin/coherent_search        FILE.fft|FILE.dat [more inputs] [options]
bin/coherent_search_list   FILE.dat.list [options]
```

The top-level `make GPU=1` forwards the variable to `src/Makefile`.  If you
switch between `GPU=0` and `GPU=1`, run `make clean` first (make does not track
compiler flags).

`GPU=1` needs the CUDA toolkit (`nvcc`, `cufft`, `cudart`); the default is
`CUDA_DIR=/usr/local/cuda`.  `-arch=native` is used, override with
`CUDA_ARCH=-arch=sm_XX` for cross-building.  On a machine without CUDA use the
default `GPU=0`; the binary still accepts `--gpu` but warns and runs on the CPU.

Requirements: a C11 compiler (GCC recommended), OpenMP, FFTW3 (float and double),
and `-march=native` for the AVX2 kernels (there are portable fallbacks).  The
supplied `bashrc` puts the PSRSOFT FFTW headers and libraries on the include and
library paths.  `src/Makefile` accepts `FFTW_INC=` / `FFTW_LIB=` overrides.
Loop induction variables are declared outside the `for` header (C89 style) for
compatibility with older compilers; the sources also parse under GCC's
`-std=gnu89` mode.

## Usage

```sh
bin/coherent_search FILE.fft|FILE.dat [FILE2 ...] [options]
bin/coherent_search_list FILE.dat.list [options]
```

The command line mirrors the Julia program:

```
-t, --threshold X        S/N cutoff (default 8.0)
-o, --outputfilenm F     candidate output file (single input; default stdout)
    --outdir D           write per-file <basename>.cohout files into D
-n, --nharms N           harmonics to sum (default 60)
    --ncands N           reported candidates (default 100)
    --lobin N            lowest Fourier bin (default 100)
    --lofreq F           lowest frequency Hz (default 0.1)
    --hifreq F           highest fundamental Hz (default 125)
    --hidr F             bin step at the highest harmonic (default 0.5)
    --m N                interpolation kernel bins, even (default 16)
    --precision P        f32 (default) or f64 profile stage
    --sigma S            analytic (default) or measured noise scale
    --blocksize N        trials per chunk (default 2048)
    --maxdecim N         max harmonic-decimation factor (default 6)
    --drtol F            duplicate-collapse tolerance in bins (default 1)
    --numharm N          harmonic-collapse depth (default 16)
    --noremove           keep near-identical candidates
    --noharmremove       keep harmonically-related candidates
    --progressbar        bar instead of the text percentage
    --noprogress         no progress meter
-v, --verbose            print the trial grid and plan details
    --metricstats        write per-block/per-decimation metric statistics
    --normalize          two-pass adaptive threshold (metric in sigma units)
    --nowisdom           do not use the FFTW wisdom cache
    --wisdomfile F       FFTW wisdom path
    --prime-wisdom       build/save all FFTW plans (f32 and f64) and exit
    --threads N          OpenMP threads per process
    --gpu [N]            run on CUDA device N (default 0; needs make GPU=1)
    --plot               write candidate profile plots (SVG pages)
    --plotstem F         plot output stem
    --plotcols N         plot grid columns per page (default 3)
    --plotrows N         plot grid rows per page (default 5)
    --plot-candidates F  re-plot from a saved candidate file (no re-search)
    --zaplist F          zap frequencies/widths listed in F (PRESTO zaplist)
    --baryv V            v/c for non-B zaplist entries (default 0)
```

`--gpu a` (or `--gpu=a`) selects device `a`, numbered from 0 as in
`nvidia-smi`.  Without a number it uses device 0.  The GPU path defaults to
`--blocksize 65536`; the CPU path to 2048.  GPU runs require `--sigma analytic`
(the default) and do not support `--normalize` / `--metricstats` (the Julia
GPU path has the same restrictions).  A non-rational `--hidr` falls back to the
CPU automatically, as does a `GPU=0` build.

CPU threads are controlled with `--threads N` or `OMP_NUM_THREADS`.  **The
default is 1**, because the intended CPU deployment is one process per
`.dat`/`.fft` with as many concurrent processes as cores:

```sh
coherent_search file.dat
python/parallel_search.py -j 16 -t 1 --outdir cands *.dat -- --maxdecim 16
```

For a single command that should use many cores, pass `--threads N` (or set
`OMP_NUM_THREADS`).  On this 16-core/32-thread host, 14 threads is a good value:

```sh
coherent_search --threads 14 file.fft
coherent_search_list --threads 14 --outdir cands dms.dat.list
```

The prefetch pipeline (realfft/rednoise overlapped with the search) runs only
when the CPU process has more than one search thread; on the GPU it always runs
because the host is idle while kernels execute.

Examples:

```sh
# Julia's test invocation (using the bundled DM380 red-noise file)
bin/coherent_search --numharm 32 -t 5.5 --maxdecim 16 \
  G53.48+0.51_20200531_snapshot-M05-P3-c2048b1-0_DM380.00_red.fft

# GPU run on device 0
bin/coherent_search --gpu 0 -t 8 file.fft

# Raw time series: realfft + rednoise are done in memory
bin/coherent_search -t 5.5 --maxdecim 16 observation.dat

# A DM sweep from a list; produces <stem>.cohout per input
bin/coherent_search_list -t 5.5 --maxdecim 16 --outdir results files.list

# Multi-file sweep of existing FFTs
bin/coherent_search -t 7 --outdir results file1.fft file2.fft

# Candidate profile plots (pages <stem>_01.svg, ...)
bin/coherent_search --plot --plotcols 3 --plotrows 5 -o cands.cohout obs.fft

# Zap known birdies from a PRESTO-format zaplist
bin/coherent_search --zaplist Lband.zaplist -t 5.5 obs.fft
bin/coherent_search_list --zaplist Lband.zaplist --outdir cands dms.dat.list
```

### Zaplists (birdie removal)

`--zaplist FILE` suppresses known interference frequencies using PRESTO's
`accelsearch -zaplist` rules.  Each non-comment line holds `frequency_Hz
width_Hz`; a line beginning with `B` is a barycentric frequency (left
uncorrected), while other lines are multiplied by `(1 + --baryv)` (default
`--baryv 0`).  The intervals are sorted by low bin, clamped below Nyquist, and
each bin range `[floor(lobin), ceil(hibin))` is replaced by a constant amplitude
derived from the 200-bin median powers immediately before and after the bird,
with zero phase -- exactly PRESTO's in-memory `zapbirds()`.

File format (the bundled example is `test_data/test/Lband.zaplist`):

```
# lines starting with '#' are comments
# frequency_Hz   width_Hz
50.0              0.2        # topocentric: corrected by (1 + --baryv)
1000.0            2.0
B 100.0           1.0        # barycentric ('B'): used as-is
```

Both `coherent_search` and `coherent_search_list` accept `--zaplist`; the same
list is parsed once and reused for every input.  It also works with `--gpu`
(zapping happens in host memory before the spectrum is uploaded), with `.dat`
input (after the in-memory realfft+rednoise), and with `--plot`/`--plot-candidates`.

The zapping is applied after `.fft`/`.dat` loading and before searching, so it
affects the search, duty-cycle measurement and plots consistently.  `.fft`
amplitudes are mapped `MAP_PRIVATE`, so the input file on disk is untouched.
`coherent_search_list` parses the list once and applies it to every input.
The resulting zapped FFT is byte-identical to PRESTO's `zapbirds -zap` output,
and the candidate files are identical.

### Plotting

`--plot` reconstructs each reported candidate's pulse profile with the same
wide-kernel `m=64` path Julia uses, rotates it so the peak sits at phase 0.5, and
lays it out in a `--plotcols × --plotrows` grid per US-Letter page.  Each panel
carries the rank, S/N, harmonic count (and decimation `k`), frequency and period;
the page header carries the file, object, DM, `T`, `N`, search configuration and
threshold.  Pages are written as self-contained SVG (`<stem>_01.svg`, ...), so no
graphics library is needed.  `--plotstem` overrides the stem; otherwise it is
derived from the candidate filename or the input filename, exactly as in Julia.

`--plot-candidates CANDS` is the standalone re-plotter (Julia's
`bin/plot_candidates.jl`): give it one input file and a saved `.cohout`, and it
reconstructs the profiles without running the search.  Without `-n` it folds at
each candidate's detection harmonic count, as the Julia standalone does; with
`-n` it folds at that depth.

### Python tools

The original `bin/` Python programs are provided in `python/` and no longer
invoke Julia:

```sh
# Cross-observation / DM sifting (output byte-identical to the Julia copy)
python/sift_candidates.py --min-snr 5.5 --html sift.html cands/

# Drop RFI-like signals and those with no neighbouring-DM detection
python/sift_candidates.py -noRFI -DMs 2 --html sift.html cands/

# Partition a file list over processes/devices
python/parallel_search.py -j 8 -t 1 *.fft -- --threshold 8
python/parallel_search.py --gpu -j 1 --gpus 0 *.dat -- --maxdecim 16
```

In `sift_candidates.py`, `-noRFI` drops signals whose class is `RFI-like`, and
`-DMs n` with `n >= 2` drops signals carrying the `no-DM-neighbour` flag
(`n < 2` disables that filter).  Both filters also apply to the HTML report.

`parallel_search.py -t N` sets `OMP_NUM_THREADS=N` for each job, and `--gpu`
assigns devices through `CUDA_VISIBLE_DEVICES` while passing `--gpu 0` to each
child.  Inputs may be `.fft` or `.dat`; single-file jobs are forced to write a
`.cohout` rather than stdout.  Without `-j`, the job count defaults to the
physical-core count (respecting CPU affinity), not the logical count.

### `.dat` input and `coherent_search_list`

For a `.dat` input the program reads `<name>.inf` for `N`/`dt`, then applies the
same steps as

```sh
realfft name.dat     # PRESTO's in-core real FFT
rednoise name.fft    # default startwidth=6, endwidth=100, endfreq=6 Hz
```

entirely in memory.  `rednoise`'s DC/Nyquist bin is set to `(1,0)` exactly as
PRESTO does.  No `.fft` or `_red.inf` file is created.  Candidate files for a
`.dat` are named `<stem>.cohout` (beside the input, or in `--outdir`), exactly
as for a `.fft` input with the same stem.

A list file contains one `.dat` (or `.fft`) path per line, relative to the list
file's directory; blank lines and `#` comments are skipped.  Runs whose inputs
would resolve to the same candidate filename are rejected (for example a
`foo.dat` and a `foo.fft`, or equal basenames collected with `--outdir`),
rather than silently overwriting one result.  Exit status is 0 on success, 1 if
an input cannot be opened/prepared or a candidate file cannot be written, and 2
for command-line or list errors.  `coherent_search_list` keeps one `DatPipe` (realfft plan + rednoise buffers), one `SearchCache`
(harmonic tables, chunk starts, workspaces, FFTW plans) and one CUDA session
(device buffers, cuFFT plans, uploaded interpolation tables) alive for the whole
list, so only the file read, FFT/rednoise and the search itself are repeated per
DM.

## What is implemented

- PRESTO `.inf` / `.fft` readers, mmap of the amplitudes.
- In-memory `.dat` -> `realfft` -> `rednoise` (faithful ports of PRESTO's
  algorithms, including its quickselect median, `invsqrtf` and block/slope
  bookkeeping; the FFTW plans are built the same way PRESTO builds them).
- `coherent_search_list`: a DM-list driver that reuses every file-independent
  allocation and plan across the list.
- Exact rational trial grid (`lodr = hidr/nharms`), per-harmonic phase tables
  and the trials-axis (`V = 32`) interpolation kernel of the Julia production
  path, including the `Float32` weights and the non-rational fallback.
- Batched complex-to-real FFTs (FFTW) for the base fold and for **strided**
  harmonic-decimation views `k = 2..maxdecim`.
- An optional CUDA backend (`make GPU=1`, `--gpu [N]`): the same interpolation
  tables drive a warp-per-group kernel, cuFFT does the batched inverse
  transforms, and an exact per-profile boxcar kernel produces the metrics.
  The GPU is selected by index, works with f32 and f64 profile precision, and
  reproduces the Julia GPU candidate files byte for byte.
- The zero-mean, unit-L2 boxcar matched filter with the exact band-limited
  normalisation (`boxcar_shape`), the `Float32` batched gate plus `Float64`
  re-score, and the pooled MAD noise estimator (`--sigma measured`).
- Chunking with forced splits at every change of the usable harmonic set, so
  results are invariant to `--blocksize` and thread count.
- Duplicate/harmonic collapsing, rank/metric ordering, `--ncands`, and duty
  cycles measured from wide-kernel (`m=64`) reconstructed profiles.
- `--metricstats` (per-block table, per-(k, window) FAP thresholds and
  histograms), `--normalize` (two passes + `MetricNorm`), FFTW wisdom per
  precision.
- PRESTO-compatible `--zaplist` birdie zapping (accelsearch semantics), applied
  to the in-memory spectrum only; the input `.fft` file is never modified.
- Multi-file runs and the same stdout/`.cohout` formatting.

## Validation

The C binary was compared line-for-line against the Julia reference on the
bundled `G53.48+0.51 ... DM380.00_red.fft` data for:

| Case | Result |
|---|---|
| defaults | identical |
| `-t 6 -n 32 --maxdecim 6 --m 32` | identical |
| `-t 5 --maxdecim 3 --precision f64` (101 candidates) | identical |
| `--sigma measured` | identical |
| `--normalize` | identical |
| `--noremove --noharmremove` (many candidates) | identical |
| `--lobin 500 --hifreq 20`, `--maxdecim 1`, `--numharm 32` | identical |
| `--hidr 0.37` (non-rational fallback), `--m 8/64`, `-n 37`, `-n 10` | identical |
| `--blocksize 64/100/512/4096/100000`, `--ncands 5` | identical |
| f64 + measured, f64 + normalize | identical |
| `--metricstats` files (`_metricstats.txt`, `_metricfap.txt`, `_metrichist.txt`) | identical |
| 1 vs 8 vs 16 threads; 64 vs 2048 vs 100000 blocksize | identical |
| `-o`, `--outdir`, multiple input files | identical |
| all 7 bundled `.fft` files, CPU | identical to Julia CPU |
| all 7 bundled `.fft` files, `--gpu 0` | byte-identical to Julia `--gpu` |
| GPU `--blocksize 8192/16384/32768/65536/131072`, `-n 120` and `-n 240` | identical to CPU |
| GPU f64; GPU with non-rational `--hidr` (CPU fallback) | identical |
| `python/sift_candidates.py` vs the Julia `bin/sift_candidates.py` | text/stderr identical; HTML identical |
| `python/parallel_search.py` CPU jobs vs sequential `coherent_search` | identical .cohout files |
| `--prime-wisdom` f32/f64 | builds and persists both wisdom files |
| `--plot` page count/layout vs Julia `--plot` | same pages, same captions |
| `--plot-candidates` vs Julia `bin/plot_candidates.jl` | same pages/captions |
| `--zaplist` vs PRESTO `zapbirds -zap` (same `.fft`) | zapped FFT byte-identical; candidate file identical |
| `--zaplist` on `.dat` vs searching a PRESTO-zapped `_red.fft` | identical candidates |
| `--zaplist` with `.fft`/`.dat` list and GPU | single-file/list/GPU agree |
| `compute-sanitizer` memcheck and racecheck | 0 errors/hazards |
| ThreadSanitizer | not runnable in this container (`unexpected memory mapping`); ASan/UBSan and GPU racecheck are clean |
| 105-file `.dat` list, CPU vs Julia on the provided `_red.fft` files | 104 files byte-identical; 1 file one S/N differs by 0.01 (same candidate) |
| 105-file `.dat` list, GPU vs Julia `--gpu` | **all 105 files byte-identical** |
| 105-file `.dat` list, GPU vs CPU | same candidates/frequencies; 49 files have 1-2 lines differing in the last printed digit or a 0.01 S/N (cuFFT vs FFTW rounding and greedy-collapse ties; Julia's own CPU/GPU shows the same class of difference) |

AddressSanitizer + UndefinedBehaviorSanitizer builds run clean, as does
`compute-sanitizer --tool memcheck` on the CUDA backend.

Raw profile metrics can differ in the last bits between FFTW builds (Julia ships
FFTW 3.3.12 through `FFTW_jll`; this machine's PSRSOFT library is 3.3.10).
That does not change any candidate, block statistic, FAP threshold or histogram
on the tests above.

## Performance

Benchmark host: AMD Ryzen 9 9950X (32 logical CPUs), RTX 4080 SUPER, CUDA 12,
FFTW 3.3.10, Julia 1.13.1.  Timings are wall clock, with `--noprogress`, and
write their candidate files to `/tmp`.

Heavy configuration: `-n 120 --maxdecim 16 -t 8` on a single 12.4 MB,
T = 305.3 s FFT: 4,548 chunks, 9.15 M trial fundamentals.  Wall clock after
the FFTW wisdom cache is warm (no GPU used):

| threads | C | Julia |
|---:|---:|---:|
| 1 | 8.56 s | 9.19 s |
| 4 | 2.31 s | 2.90 s |
| 8 | **1.63 s** | 2.35 s |
| 16 | 2.20 s | 2.94 s |
| 32 | 3.48 s | 4.24 s |

105-file DM sweep from `test_data/test` (`--numharm 32 -t 5.5 --maxdecim 16`).
C rows use the 105 `.dat` files and include the internal realfft/rednoise;
Julia rows use the corresponding `_red.fft` files (Julia has no in-memory
`.dat` path):

| run | wall | notes |
|---|---:|---|
| Single `.fft`, C default (1 thread) | 2.05 s | one core per process |
| Single `.fft`, C `--threads 16` | **0.27 s** | Julia `-t auto` 1.20 s; Julia `-t 16` 0.92 s |
| C CPU, 105-file list, default (1 thread) | 216 s (3.6 min) | intended for batch mode |
| C CPU, 105-file list, `--threads 14` | **23.7 s** | ~200 MB RSS |
| C CPU, 105-file list, `--threads 8` | 30.2 s | memory-bandwidth limited |
| C CPU, 105-file list, 12 processes × `-t 1` | 30.1 s | ~65 MB RSS each |
| C CPU, 105-file list, 16 processes × `-t 1` | 38.1 s | memory-bandwidth bound |
| C GPU, 105-file list | **7.05 s** | ~250 MB RSS, one shared context |
| Julia CPU, single file / 105-file list | 1.20 s / 62.6 s | Julia `-t auto`, includes startup |
| Julia GPU, 105-file list | 19.5 s | includes CUDA.jl loading |

The default one thread keeps a process-per-file scheduler predictable.
Multi-process runs are limited by memory bandwidth on this host: 12
single-threaded processes are faster than 16, and both are slower than one
14-thread process for this particular 105-file sweep.  Use `--threads N` (or
`OMP_NUM_THREADS`) when a single process should use more cores.

The GPU default is `--blocksize 65536`.  `--blocksize 32768` is about 4% faster
on this list, but chunk boundaries differ from Julia's 65536 default, which can
move a frequency in its 12th decimal or resolve an exact metric tie differently;
the candidate set is still equivalent.  Keep 65536 for byte-identical
comparisons, lower it for speed.

Per-file `realfft`+`rednoise` in memory costs about 0.05 s.  When a CPU process
has more than one search thread (or when running on the GPU), the list driver
runs two prefetch producer threads over a four-slot buffer, hiding that work
behind the search: the 105-file list takes 23.7 s with 14 CPU threads and
7.05 s on the GPU.

Further optimisations in this version:

- GPU metric kernels append only above-threshold candidates to compact record
  buffers with atomics, so the host no longer scans/copies ~73 million metrics
  per file (105-file GPU run: 18.4 -> 7.05 s);
- candidate records are written straight into mapped pinned host memory; each
  GPU chunk uses one blocking count copy as its completion point, with no
  explicit stream synchronization and no record device-to-host copies;
- the finished CPU spectrum is uploaded once per file (`cudaMemcpy` of the
  whole `ft->amps`); interpolation reads `d_amps` directly, with no segmented
  uploads and no device-to-device copy of the FFT data;
- CUDA/cuFFT return-value checks are not used in the compute path; device
  discovery still validates that the requested GPU exists;
- all decimation gathers are one packed kernel launch per chunk, and the
  active-harmonic state is one 16-byte-per-entry copy instead of three;
- boxcar shapes are cached per decimation and only recomputed/copied when the
  filled-harmonic count changes;
- the CPU interpolator zeroes only the DC row and harmonics that are actually
  unfilled, instead of the whole `(nharms+1) x Nprof` stack each chunk;
- a process-wide FFTW planner mutex makes the prefetch and search threads safe
  (plan execution remains parallel).

A hand-written AVX-512 assembly version of the boxcar scan was also implemented
and benchmarked.  It produced bit-identical results but ran ~17% slower than the
compiler's intrinsics (22.8 s vs 26.8 s on the 105-file list), so it was removed:
the C kernels already compile to the expected `zmm`/`ymm` loads, `vmaxps` and
FMA sequences and GCC schedules them better than the hand-written loop.

GPU (same 12.4 MB FFT, `-t 8`, default 65536-trial chunks; CUDA context and
cuFFT plan setup included):

| workload | CPU 8 threads | GPU | Julia GPU (end to end) |
|---|---:|---:|---:|
| `-n 60 --maxdecim 16` | 0.29 s | **0.12 s** | 8.4 s |
| `-n 120 --maxdecim 16` | 1.63 s | **0.52 s** | 8.4 s |
| `-n 240 --maxdecim 16` | 9.57 s | **2.36 s** | — |

The C GPU fixed setup is about 0.1 s here because this host runs a CUDA MPS
server; earlier cold-context runs included about 0.8 s and were slower than the
CPU for `-n 60`.  Julia's figures are dominated by CUDA.jl loading (~8 s); on
this MPS-enabled host Julia was run with `JULIA_CUDA_USE_COMPAT=false` so it
selects the system driver.

Single-thread phase totals (heavy config): interpolation 1.30 s, base inverse
FFT 0.87 s, base metric 0.79 s, decimated inverse FFTs 3.51 s, decimated metrics
1.68 s — against Julia's 1.17 / 1.05 / 0.83 / 3.63 / 1.81 s.

The original CPU optimisations that got there:

1. explicit AVX2 FMA accumulators for the interpolation group kernel (eight live
   vectors, avoiding GCC's AVX-512 register-pressure choice);
2. an explicit AVX2 boxcar scan — GCC's auto-vectorised form was 2.4-4.8x slower;
3. a fused transpose+prefix-sum pass that writes already-transposed prefix sums
   into `psT`, removing the intermediate `(B, nbins)` tile and its memory traffic.

## Known limitations

- `--gpu` requires a `GPU=1` build.  With `GPU=0`, or on a machine with no
  accessible CUDA device, it warns and runs the CPU path.
- GPU S/N values differ from the CPU path by cuFFT-vs-FFTW rounding (~1e-6 in
  Float32).  On the 105-file DM list this changes the last printed digit for a
  few candidates and, because the harmonic collapse is greedy on nearly tied
  metrics, occasionally which member of a tie survives — the same effect Julia's
  own GPU/CPU comparison shows.  Our GPU output is byte-identical to Julia's GPU
  output on all 105 files.
- For `.dat` input, the internal realfft/rednoise reproduces PRESTO's algorithms
  but not bit-for-bit: FFTW's plan/rounding differs in the last bits and
  rednoise's `-Ofast` arithmetic cannot be replicated exactly under the
  project-wide no-contraction build.  Candidate frequencies and periods agree
  to ~1e-12; on the 105-file list one candidate's S/N prints 5.53 instead of
  Julia's 5.54.
- `--plot` writes SVG rather than Julia's PNG (CairoMakie).  The content,
  layout, rotation and captions match; SVG needs no extra libraries and can be
  converted with a browser, Inkscape or `rsvg-convert` if PNG is required.
- The default GPU `--blocksize` (65536) matches Julia for byte-identical output;
  `--blocksize 32768` is faster but may move a chunk-boundary frequency by one
  ulp or break an exact metric tie differently in post-processing.
- GPU `--precision f64` doubles the metric kernel's shared-memory footprint.
  For very large `--nharms` it can exceed the device's opt-in shared-memory
  limit; the session then exits with an explicit error.  f32 (the default) and
  f64 up to a few hundred harmonics fit on current cards.
- `bin/toy_coherent_search.jl` (the deliberately slow, unoptimised teaching
  reference) has no C counterpart; the production C path is the port.
- The standalone plotter is `coherent_search --plot-candidates`, not a separate
  binary.
- `--noplot` is accepted and ignored; plotting is off unless `--plot` is given.

## Layout

```
include/coherent_search.h   types and the public surface
src/fileio.c                .inf / .fft readers
src/interp.c                rational grid, phase/group tables, chunk filling,
                            harmonic-change boundaries, exact point kernel
src/boxcar.c                medians, width banks, noise scales, scalar scan,
                            AVX2 batched gate + fused prefix transpose
src/search.c                workspaces, FFTW plans, decimation, threaded chunk
                            loop, metric statistics, normalisation, wisdom
src/candidate.c             candidate lists/collapsing, profile reconstruction,
                            duty cycle, output formatting
src/datpipe.c               in-memory realfft + rednoise (.dat input)
src/cli.c                   argument parsing, list mode, --metricstats
src/plot.c                  dependency-free SVG candidate-profile pages
src/cuda.cu                 CUDA backend (interp, gather, cuFFT, metric)
src/gpu_stub.c              no-CUDA stubs for GPU=0 builds
include/cuda.cuh            CUDA-internal declarations
src/CoherentSearch.c        main() for coherent_search
src/list_main.c             main() for coherent_search_list
python/sift_candidates.py   cross-observation/DM candidate sifting (no Julia)
python/parallel_search.py   split a file list over CPU jobs or CUDA devices
bin/sift_candidates.py      symlink to python/sift_candidates.py
bin/parallel_search.py      symlink to python/parallel_search.py
src/Makefile                build (GPU=0/1)
```
