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

For searching **raw** observations directly, the additional front end
`prep_coherent_search` performs PRESTO `prepsubband` de-dispersion in memory and
then runs the same search without writing any intermediate `.dat` files.  See
[Raw data](#raw-data-prep_coherent_search).

## AI assistance

This C port, including the CUDA backend, build files, and the Julia-independent
Python tools, was rewritten with assistance from **DeepSeek**.

## References

- Original Julia implementation:
  <https://github.com/scottransom/CoherentSearch.jl>
- Paper: <https://arxiv.org/abs/2610.24287>

## Build

```sh
source ~/bashrc
cd CoherentSearch_C
make            # CPU-only binary (GPU=0, the default)
make GPU=1      # also build the CUDA/cuFFT backend
make LTO=1      # add GCC link-time optimization (same results, ~1-3% faster)
make GPU=1 LTO=1
make GPU=1 CUDA_ARCH=-arch=sm_89   # explicit target; default is -arch=native
make GPU=1 NVCC_EXTRA="-Xptxas -dlcm=cg"   # optional ptxas/codegen flags
make VECTOR_WIDTH=auto   # 256 on AMD AVX-512, 512 on Intel AVX-512
make VECTOR_WIDTH=256    # or force 128/256/512; default native
```

This builds the main programs in either configuration:

```
bin/coherent_search              FILE.fft|FILE.dat [more inputs] [options]
bin/coherent_search_list         FILE.dat.list [options]
```

and the raw-data front end in **both** `GPU=0` and `GPU=1` builds:

```
bin/prep_coherent_search         <prepsubband opts> ++ <search opts>
```

`GPU=0` builds the prep program with CPU-only dedispersion and no CUDA
linkage; `GPU=1` also enables `-cuda` dedispersion.

The top-level `make GPU=1` forwards the variable to `src/Makefile`.  If you
switch between `GPU=0` and `GPU=1`, run `make clean` first (make does not track
compiler flags).

`GPU=1` needs the CUDA toolkit (`nvcc`, `cufft`, `cudart`); the default is
`CUDA_DIR=/usr/local/cuda`.  `-arch=native` is used, override with
`CUDA_ARCH=-arch=sm_XX` for cross-building.  On a machine without CUDA use the
default `GPU=0`; the binary still accepts `--gpu` but warns and runs on the CPU.

Requirements: a C11 compiler (GCC recommended), OpenMP, FFTW3 (float and
double), and `-march=native` for the AVX2 kernels (there are portable
fallbacks).  The prep front end additionally needs cfitsio.  The supplied
`bashrc` puts the PSRSOFT FFTW headers and libraries on the include and library
paths.  `src/Makefile` accepts `FFTW_INC=` / `FFTW_LIB=` overrides.  The PRESTO
reader/dedispersion sources are vendored under `third_party/presto/`, so no
external PRESTO installation or `libpresto` is needed.
Loop induction variables are declared outside the `for` header (C89 style) for
compatibility with older compilers; the sources also parse under GCC's
`-std=gnu89` mode.

## Usage

```sh
bin/coherent_search FILE.fft|FILE.dat [FILE2 ...] [options]
bin/coherent_search_list FILE.dat.list [options]

# raw observations: prepsubband options, then ++, then search options
bin/prep_coherent_search <prepsubband options> -o OUT RAW... ++ <search options>
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
    --progressbar        accepted for compatibility (progress output is off)
    --noprogress         accepted for compatibility (progress output is off)
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

# Drop RFI-like, harmonic, and no-neighbour-DM signals
python/sift_candidates.py -noRFI -noharm -DMs 2 --html sift.html cands/

# Partition a file list over processes/devices
python/parallel_search.py -j 8 -t 1 *.fft -- --threshold 8
python/parallel_search.py --gpu -j 1 --gpus 0 *.dat -- --maxdecim 16
```

In `sift_candidates.py`, `-noRFI` drops signals whose class is `RFI-like`,
`-noharm` drops signals classified as `harmonic`, and `-DMs n` with `n >= 2`
drops signals carrying the `no-DM-neighbour` flag (`n < 2` disables that
filter).  All filters also apply to the HTML report.

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

### Raw data: `prep_coherent_search`

`prep_coherent_search` integrates PRESTO's `prepsubband` de-dispersion.
The needed `prepsubband` reader/dedispersion sources and headers are vendored
under `third_party/presto/`; no external PRESTO installation or `libpresto` is
required.  cfitsio and FFTW are always external; CUDA is needed only for a
`GPU=1` build.

It accepts the normal `prepsubband` options (`-nsub`, `-downsamp`, `-lodm`,
`-dmstep`, `-numdms`, `-zerodm`, `-mask`, `-cache`, `-numout`, barycentering,
etc.).  The de-dispersed time series is kept in memory; the ordinary
realfft/rednoise and coherent search then run on that memory.  Only the
`<outprefix>_DM<dm>.inf` files and `.cohout` candidate tables are written ---
**no `.dat` file is created**.

Run `prep_coherent_search -h` for the combined help: it prints the complete
PRESTO `prepsubband` option list followed by the complete `coherent_search`
search-option list.  The same combined help is printed by
`prep_coherent_search ... ++ --help` without running a search.

Raw input formats:

- **PSRFITS** (`.fits`, `.sf`); `-psrfits` optional, suffix auto-detected.
- **SIGPROC filterbank** (`.fil`, `.fb`); `-filterbank` optional, suffix
  auto-detected (`identify_psrdatatype -> SIGPROCFB -> read_filterbank_files`).
  Tested end-to-end with a synthetic 8-bit filterbank.
- **Old PRESTO subbands** (`*.sub00` etc., multiple input files).
- **PRESTO cache** (`-cache`, reading `<fits>_cac` or `<fits>_cac0`); the
  original raw file is still supplied for its header.

SCAMP/BPP/WAPP/SPIGOT are recognized by PRESTO's identifier but the vendored
reader intentionally exits.  `.dat`, `.sdat` and event lists are not raw
dedispersion inputs; use `coherent_search` for an existing `.dat`.

Search options follow a literal `++` separator and are the same options as
`coherent_search` (`-t`, `-n`, `--maxdecim`, `--blocksize`, `--lofreq`,
`--outdir`, `--threads`, `--gpu`, `--zaplist`, `--precision`, `--sigma`, ...):

```sh
prep_coherent_search <prepsubband options> -o OUTPREFIX [RAW...] \
    ++ <search options> [RAW...]
```

Raw input files may be placed on either side of `++`; non-option names after
`++` are appended to the `prepsubband` input list.  When `--gpu N` is present
after `++`, the de-dispersion and the search both use CUDA device `N`; the
device is selected once and the context is reused.  An explicit `-cuda M`
on the preparation side is accepted only when `M == N`.

For example, replacing one `prepsubband` invocation from a DM-generation
script:

```sh
prep_coherent_search -numout 851968 -nsub 224 -lodm 1002.50 -dmstep 0.50 \
    -numdms 210 -downsamp 8 -psrfits -zerodm -cache \
    -mask OBS_rfifind.mask -o OUTPREFIX OBS.fits \
    ++ -t 6.0 -n 60 --maxdecim 16 --gpu 0 --blocksize 8192 --lofreq 0.03 \
       --noprogress
```

A filterbank input uses the same form (the suffix is enough):

```sh
prep_coherent_search -numout 65536 -nsub 4 -lodm 10 -dmstep 1 -numdms 1 \
    -downsamp 1 -nobary -o OUT test.fil \
    ++ -t 5.0 -n 30 --maxdecim 4 --gpu 0 --lofreq 0.1
```

This produces `OUTPREFIX_DM1002.50.inf`, `OUTPREFIX_DM1003.00.inf`, ... and
the matching `.cohout` candidate files; there is no `.dat` anywhere.  Use
`--outdir` to place the `.cohout` files elsewhere.  Omit `--gpu` for CPU
de-dispersion and a CPU search.  The prep programs are built in both `GPU=0`
and `GPU=1` modes: `GPU=0` uses CPU de-dispersion and links no CUDA or
external PRESTO library; `GPU=1` also permits GPU de-dispersion, enabled
automatically by `--gpu N` after `++` or explicitly with `-cuda N` before
`++`.

The `-cache` path reads the PRESTO `_cac`/`_cac0` file; with the same
`prepsubband` options the in-memory time series was verified bit-identical to
the `.dat` written by the reference `prepsubband`, and the resulting `.cohout`
was identical to searching that `.dat`.  `-sub` (subband output) and
`-outlist` are not meaningful here; `-sub` is rejected with an error.

The in-memory path was compared directly with writing `.dat` and then running
`coherent_search_list` on those files.  On a short-N 10-DM run the candidate
files were byte-identical for disk vs memory (CPU and GPU): disk total
8.49 s CPU / 3.07 s GPU, memory 8.58 s / 3.09 s, while memory avoided 33 MB of
`.dat`.  On a long-N 10-DM run the disk totals were 7.87 s CPU / 2.57 s GPU and
the memory totals 8.40 s / 3.03 s; 9 of 10 candidate files were byte-identical,
and one had S/N 6.19 vs 6.18 at the same frequency (a barycentric bin-boundary
rounding difference), while memory avoided 260 MB of `.dat`.  The small-set
timing is nearly identical because written `.dat` files are still in the page
cache; the in-memory path mainly removes the 221 GB of intermediate `.dat`
storage for the full 157-command sweep.

Practical checks on the 11.6 ms dataset (157-command generation script):
a short-N command (`numout=851968`, `nsub=224`, `downsamp=8`) run for 10 DMs
took 8.5 s with CPU search and 3.1 s with GPU search; a long-N command
(`numout=6815744`, `nsub=32`, `downsamp=1`) run for 2 DMs took 2.8 s / 1.7 s.
`-cache` vs direct FITS gave identical `.cohout` files for the tested DMs,
the log confirmed `Read cache file ..._cac0`, and no `.dat` or `.dat.list`
files were created.

## What is implemented

- PRESTO `.inf` / `.fft` readers, mmap of the amplitudes.
- In-memory `.dat` -> `realfft` -> `rednoise` (faithful ports of PRESTO's
  algorithms, including its quickselect median, `invsqrtf` and block/slope
  bookkeeping; the FFTW plans are built the same way PRESTO builds them).
- `coherent_search_list`: a DM-list driver that reuses every file-independent
  allocation and plan across the list.
- `prep_coherent_search`: PRESTO `prepsubband`
  front end that dedisperses raw PSRFITS/filterbank/cache data into memory,
  write only `.inf` files, and search the in-memory series on CPU or GPU.
- Exact rational trial grid (`lodr = hidr/nharms`), per-harmonic phase tables
  and the trials-axis (`V = 32`) interpolation kernel of the Julia production
  path, including the `Float32` weights and the non-rational fallback.
- Batched complex-to-real FFTs (FFTW) for the base fold and for **strided**
  harmonic-decimation views `k = 2..maxdecim`; when `k` divides `nharms` the
  decimated profile is instead an exact fold of the base profile (no FFT).
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
| 8 | **1.57 s** | 2.35 s |
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
| C CPU, 105-file list, `--threads 14` | **23.3 s** | ~200 MB RSS |
| C CPU, 105-file list, `--threads 8` | 29.8 s | memory-bandwidth limited |
| C CPU, 105-file list, 12 processes × `-t 1` | 30.1 s | ~65 MB RSS each |
| C CPU, 105-file list, 16 processes × `-t 1` | 36.1 s | memory-bandwidth bound |
| C GPU, 105-file list | **5.7 s** | ~250 MB RSS, one shared context |
| Julia CPU, single file / 105-file list | 1.20 s / 62.6 s | Julia `-t auto`, includes startup |
| Julia GPU, 105-file list | 19.5 s | includes CUDA.jl loading |

The default one thread keeps a process-per-file scheduler predictable.
Multi-process runs are limited by memory bandwidth on this host: 12
single-threaded processes are faster than 16, and both are slower than one
14-thread process for this particular 105-file sweep.  Use `--threads N` (or
`OMP_NUM_THREADS`) when a single process should use more cores.

The GPU default is `--blocksize 65536`.  `--blocksize 32768` is about 19%
faster on the `-n 240` workload (2.08 -> 1.69 s), but chunk boundaries differ
from Julia's 65536 default, which can move a frequency in its 12th decimal or
resolve an exact metric tie differently;
the candidate set is still equivalent.  Keep 65536 for byte-identical
comparisons, lower it for speed.

On the 1.6 ms dataset, a single process is flat between 32768 and 65536
(6.2 s per 100-file list); below that, launch overhead wins: 16384 takes 7.2 s,
8192 10.0 s, and 4096 15.9 s.  Small chunks are useful only when several MPS
clients run concurrently (see the multi-process section).

A single GPU process uses only about 0.6 GB of VRAM at 65536; using more VRAM
by enlarging the chunk is counterproductive.  On the 1.6 ms dataset, a
100-file list took 6.2 s at 65536, 11.4 s at 131072, 15.4 s at 262144, and
23.9 s at 524288.  Persisting the amplitude array in L2 also slowed it
(6.9 s).  The limit is L2/bandwidth and per-chunk kernel efficiency, not VRAM
capacity; 65536 is the single-process optimum here.  Extra VRAM is only useful
for several MPS clients with smaller per-client chunks.

The CPU default is `--blocksize 2048`, matching Julia's CPU chunking.  On this
host `--blocksize 1024` is about 15% faster on the 105-file list (19.8 s vs
23.3 s with 14 threads) and about 24% faster on the `-n 120` 8-thread case
(1.2 s vs 1.57 s).  As with the GPU block size, this changes chunk boundaries and
can move a printed frequency in its last digit or alter a tie, so 2048 remains
the default for byte-identical output.

Per-file `realfft`+`rednoise` in memory costs about 0.05 s.  When a CPU process
has more than one search thread (or when running on the GPU), the list driver
runs two prefetch producer threads over a four-slot buffer, hiding that work
behind the search: the 105-file list takes 23.3 s with 14 CPU threads and
5.7 s on the GPU.

Further optimisations in this version:

- GPU metric kernels append only above-threshold candidates to compact record
  buffers with atomics, so the host no longer scans/copies ~73 million metrics
  per file (105-file GPU run: 18.4 -> 5.7 s);
- candidate records are written straight into mapped pinned host memory; each
  GPU chunk uses one blocking count copy as its completion point, with no
  explicit stream synchronization and no record device-to-host copies;
- the finished CPU spectrum is uploaded once per file (`cudaMemcpy` of the
  whole `ft->amps`); interpolation reads `d_amps` directly, with no segmented
  uploads and no device-to-device copy of the FFT data;
- CUDA/cuFFT return-value checks are not used in the compute path; device
  discovery still validates that the requested GPU exists;
- decimation cuFFT plans read the strided harmonic rows directly from
  `d_ftprofs` (`istride=k`, `idist=nharms+1`), eliminating the gather kernel
  and its device-to-device staging buffer; the CPU path already used the same
  zero-copy layout with FFTW (`ws->ftprofs`, `istride=k`);
- the active-harmonic state is one 16-byte-per-entry copy instead of three;
- boxcar shapes are cached per decimation and only recomputed/copied when the
  filled-harmonic count changes;
- the CPU interpolator zeroes only the DC row and harmonics that are actually
  unfilled, instead of the whole `(nharms+1) x Nprof` stack each chunk;
- a process-wide FFTW planner mutex makes the prefetch and search threads safe
  (plan execution remains parallel);
- the `.dat` pipeline creates the inverse FFTW plan only to reproduce PRESTO's
  forward-plan choice, then releases it immediately (it is never executed).

A second optimisation round (all bit-identical to the previous output; every
timing below is the 50-file `G29.02` DM sweep, `-n 32 -t 6 --maxdecim 16`,
single-threaded host unless noted):

- the CPU interpolator now writes its rotated values into an **h-major
  spectrum buffer** (element `(h, j)` at `h*Nprof + j`): one contiguous 256 B
  store per warp-equivalent group instead of 32 complex scatter writes with a
  976 B stride, and the DC/inactive-harmonic zero fill becomes one `memset`
  per row.  FFTW keeps bit-identical arithmetic with the strided input
  (`istride=Nprof`, `idist=1`), about 6% slower per transform than the
  trial-major layout but far cheaper overall than the stores it replaces
  (fill stage 0.41 -> 0.29 s per file; whole CPU search -13%);
- the batched boxcar gate is **tiled by 32 profiles**: the transposed prefix
  sums plus the 32 source columns (~31 KB) stay resident in L1D while the
  ladder widths are scanned, instead of streaming the whole 65+ KB batch
  through L1D for every width;
- the per-trial candidate scan of both the base pass and every decimation is
  AVX-512 vectorised with the exact per-element arithmetic (exact
  integer->double conversion, one multiply, one add; hits replayed in
  ascending trial order), and the `nvalid` Nyquist count in each decimation
  pass is a bisection on the same monotone predicate instead of a linear
  scan;
- the GPU session executes **consecutive chunks on three CUDA streams** with
  double-buffered spectra, profiles, decimated profiles, candidate records,
  counts and upload staging per stream: kernel launches, the analytic-shape
  updates and the candidate readback of chunk *c* overlap chunk *c+1*'s
  kernels, and the metric kernels co-execute (they were previously limited to
  2 resident warps per SM at `--blocksize 8192`).  Counts return through an
  async pinned copy plus one event per chunk -- no blocking copy, no device
  idle between chunks (GPU list run 3.99 -> 2.6 s at 8192, 11.8 -> 6.7 s at
  2048);
- all harmonic plan tables are uploaded as **one packed device arena** (two
  `cudaMalloc`s and two copies instead of ~660) and the cuFFT plans are
  switched per stream with `cufftSetStream` instead of being bound to one
  stream.

Measured effect on the full 50-file sweep (single process, `OMP_NUM_THREADS=1`;
GPU rows are `--gpu 0`, block sizes as marked):

| configuration | before | after |
|---|---:|---:|
| GPU list, blocksize 8192 | 3.99 s | **2.62 s** |
| GPU list, blocksize 2048 | 11.81 s | **6.67 s** |
| `prep_coherent_search`, 8192 / 2048 | 10.7 / 18.7 s | **9.0 / 13.2 s** |
| CPU list, 8192 / 2048 | 73.7 / 73.6 s | **64.1 / 61.3 s** |
| CPU `prep_coherent_search`, 8192 | 78.6 s | **70.7 s** |

Every `.cohout` file of every configuration above is byte-identical to the
pre-change output (CPU, GPU, f32/f64, `--sigma measured`, `--normalize`,
`--noremove/--noharmremove`, other `--blocksize/--nharms/--maxdecim/--hidr`
values included).

A third round kept the byte-identical guarantee and pushed the same sweep to

| configuration | second round | third round |
|---|---:|---:|
| GPU list, blocksize 8192 | 2.62 s | **2.49 s** |
| GPU list, blocksize 2048 | 6.67 s | **6.39 s** |
| `prep_coherent_search`, 8192 / 2048 | 9.0 / 13.2 s | **8.9 / 12.7 s** |
| CPU list, 8192 / 2048 | 64.1 / 61.3 s | **63.6 / 61.9 s** |
| CPU `prep_coherent_search`, 8192 | 70.7 s | **68.9 s** |

- **GPU region cache**: the filled-harmonic pattern is constant between the
  change points the chunk table already splits on, so per region the analytic
  sigmas (base and every decimation) are computed once instead of sixteen
  times per chunk.  (Also skipping the zero-columns upload/kernel per region
  was tried and **reverted**: cuFFT's C2R destroys its input array even
  out-of-place, so the inactive/DC columns are clobbered by every exec and
  must be re-zeroed each chunk -- skipping it corrupted ~37% of the files in
  a 10850-file sweep.  The same sweep caught it: the per-stream pipeline made
  a full-scale correctness campaign worthwhile.  Each stream now also owns
  its own cuFFT plan copies, because a plan's work area is shared state and
  concurrent executions of one plan on different streams could race.)
- the **f32 gate re-scan was removed**: the batched f32 gate is bit-identical
  to the scalar per-profile filter (verified exhaustively -- 80.4 M metric
  comparisons on one observation, 0 mismatches), so re-computing candidates
  above `exactcut` was pure waste.  The f64 gate keeps its re-scan (it
  accumulates through a float `psT` and is approximate);
- the CPU interpolator's tap loop now runs **two groups per pass** (eight
  live FMA chains), halving the latency-bound iterations; each group's FMA
  order is unchanged.

Three further GPU restructurings were implemented, measured and reverted
because they lost to the three-stream overlap already in place: splitting
`fold_metric_kernel` per decimation (3.51 s vs 2.65), merging the eight
`metric_kernel` launches into one (2.97 s), and a two-group interpolation
kernel (2.67 s).

Full-scale campaign (157 lists / 10850 `.dat` files, 293 GB, same options as
the small sweep): the GPU list driver is the fastest deployment at **558 s
serial**; eight parallel list processes are identical (520-558 s -- the GPU
is saturated by one process), per-file `coherent_search` at `-P 8` costs
1145 s (about 57 ms/process startup overhead, partly amortised), and `-P 16`
is slightly worse than `-P 8`.  On the CPU (`--blocksize 2048`), `-P 16`
takes 3190 s while `-P 8` takes 2058 s: past ~8 processes the memory
bandwidth wall dominates (per-core efficiency drops to ~1/4 at 16), so
8-12 processes are the sweet spot there.  All outputs verified against the
pre-optimisation binaries (0 differing files in sampled spot checks across
the three option sets, after the cuFFT-input fix above).

A hand-written AVX-512 assembly version of the boxcar scan was also implemented
and benchmarked.  It produced bit-identical results but ran ~17% slower than the
compiler's intrinsics (22.8 s vs 26.8 s on the 105-file list), so it was removed:
the C kernels already compile to the expected `zmm`/`ymm` loads, `vmaxps` and
FMA sequences and GCC schedules them better than the hand-written loop.

GPU (same 12.4 MB FFT, `-t 8`, default 65536-trial chunks; CUDA context and
cuFFT plan setup included):

| workload | CPU 8 threads | GPU | Julia GPU (end to end) |
|---|---:|---:|---:|
| `-n 60 --maxdecim 16` | 0.29 s | **0.10 s** | 8.4 s |
| `-n 120 --maxdecim 16` | 1.57 s | **0.45 s** | 8.4 s |
| `-n 240 --maxdecim 16` | 9.43 s | **2.08 s** | — |

The C GPU fixed setup is about 0.1 s here because this host runs a CUDA MPS
server; earlier cold-context runs included about 0.8 s and were slower than the
CPU for `-n 60`.  Julia's figures are dominated by CUDA.jl loading (~8 s); on
this MPS-enabled host Julia was run with `JULIA_CUDA_USE_COMPAT=false` so it
selects the system driver.

### Multi-process GPU

A GPU is one shared device.  MPS does multiplex clients (the eight-process run
showed 99% GPU and 97% memory utilization), but at the default
`--blocksize 65536` each client streams a ~100 MB working set; eight of them
thrash L2 and saturate memory bandwidth, so aggregate throughput drops.  On a
1.6 ms dataset with 6,735 `.dat` files in 93 lists, one process using a combined
list took **400 s**, while eight concurrent `--gpu 0` list processes took
**678 s**.  On an 8-list/664-file subset the default-chunk scaling was:

| processes | wall |
|---:|---:|
| 1 | **40.0 s** |
| 2 | 58.4 s |
| 4 | 64.8 s |
| 8 | 67.8 s |

Reducing the per-process chunk size changes the working-set/cache balance.
With `--blocksize 8192`, eight MPS clients reach **31.3 s** on the same 664
files (16384: 41.7 s; 4096: 38.4 s).  On the full 6,735 files this gives
**306 s** versus 400 s for one process and 678 s for eight at 65536.  It
produces the same total number of candidate rows, but smaller chunks change the
chunk boundaries: 1,089 of 6,735 files differ in a last printed frequency digit
or a near-tie choice.  If byte-identical output is required, keep 65536 and use
one process; if throughput matters more than the last digit, use the 8192
multi-process configuration.

Other MPS experiments did not help: switching to a non-blocking stream/async
count copy, `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=12`, and an L2 streaming access
policy all left the 65536 multi-process time at 58-71 s.

Progress printing was never a measurable cost here, but it made long list runs
unreadable, so it is now gone: the per-chunk meter and all intermediate
per-file/per-pass `Info:` lines have been removed.  The programs are quiet by
default and print only errors, warnings, the `-v` plan report and the
`--metricstats` report (the latter two are the requested output).  On the same
664-file subset, `--noprogress` took 40.03 s and default progress to `/dev/null`
40.05 s; a test build with the per-chunk branch and all `Info:` prints compiled
out took 40.07 s, so the checks and formatting were never the bottleneck.
`--noprogress`/`--progressbar` are still accepted and ignored, so existing
command lines (including `test.sh`) keep working.

The one-process combined run produced the same 6,735 `.cohout` files, all
byte-identical to the eight-process 65536 outputs.  For exact output on one GPU,
combine the lists and run one process:

```sh
awk 'NF && !seen[$0]++' *.dat.list > all.dat.list
coherent_search_list --numharm 32 -t 6.0 --maxdecim 16 --gpu 0 \
    --zaplist Lband.zaplist --lofreq 0.03 all.dat.list
```

For maximum single-GPU throughput, split `all.dat.list` into about eight
balanced lists and run eight processes with `--blocksize 8192`:

```sh
split -n l/8 all.dat.list part_
for f in part_*; do
    coherent_search_list --noprogress --blocksize 8192 \
        --numharm 32 -t 6.0 --maxdecim 16 --gpu 0 \
        --zaplist Lband.zaplist --lofreq 0.03 "$f" &
done
wait
```

This is the 306 s configuration above, with the last-digit/tie caveat.

With multiple GPUs, split the list into one combined list per card and run one
process per card, setting `CUDA_VISIBLE_DEVICES` (and `--gpu 0`) for each.

Single-thread phase totals (heavy config): interpolation 1.30 s, base inverse
FFT 0.87 s, base metric 0.79 s, decimated inverse FFTs 3.51 s, decimated metrics
1.68 s — against Julia's 1.17 / 1.05 / 0.83 / 3.63 / 1.81 s.

The original CPU optimisations that got there:

1. explicit AVX2 FMA accumulators for the interpolation group kernel (eight live
   vectors, avoiding GCC's AVX-512 register-pressure choice);
2. an explicit AVX2 boxcar scan — GCC's auto-vectorised form was 2.4-4.8x slower;
3. a fused transpose+prefix-sum pass that writes already-transposed prefix sums
   into `psT`, removing the intermediate `(B, nbins)` tile and its memory traffic.

A later pass added four more changes, measured on the same 305 s observation
(`-n 60 --maxdecim 16`, one thread; the per-phase figures below were taken
with temporary timers that are not part of the shipped build):

4. an exact algebraic shortcut for the decimated inverse FFTs.  Harmonic
   subsampling by `k` is a time-domain fold, so for `k | nharms` the c2r of the
   strided stack is exactly the `k`-fold average of the full-resolution
   profile: `r_k[j] = (1/k) * sum_{q<k} r_full[j + q*n_k]`.  Those profiles are
   now produced by an AVX-512 fold of `ws->profs`; the corresponding FFTW plans
   are not created at all, and the non-divisor `k` keep FFTW.  Decimated
   inverse FFTs: 0.64 s -> 0.34 s.  `fold_decims()` then produces *all* eight
   foldable decimations in one pass over the base profile, so each 120-float
   row is read once instead of once per decimation (8 MB -> 1 MB of source
   traffic per chunk); this took the decimated transform phase to ~0.20 s and
   the single-observation wall time from 1.44 s to 1.33 s.
5. the boxcar scan loops over positions on the outside and ladder widths on the
   inside, loading `psT[p-1]` once per position instead of twice per width.
   `max` is exact and order-free, so no value changes.  Scan: 0.28 s -> 0.19 s.
6. the fused transpose+prefix pass now builds **16-wide** prefix chains: two
   8x8 AVX2 register transposes (unpack/shuffle/permute2f128, all cheap moves)
   are joined with `vinsertf32x8`, so each chain drives 16 profiles and every
   `psT` store is a full 64-byte cache line instead of half of one.  The
   prefix adds are the same adds in the same order, so `psT` stays
   bit-identical.  Transpose: 0.31 s -> 0.16 s.  Three implementations were
   benchmarked on the same shapes and all produced bit-identical `psT`
   (maxerr 0): two 8x8 + `vinsertf32x8` 60 us, the previous 8x8 pair 62 us,
   a 16x16 AVX-512 bit-swap transpose 114 us, and PRESTO's FFTW rank-0 r2r
   "transpose trick" (`fftwf_plan_guru_r2r` with zero rank, as used by
   `transpose.c` in dedispersion) 794 us.  The bit-swap version loses because
   `vpermt2ps` is expensive on Zen 5; the FFTW trick loses because it
   materialises the transposed tile (an extra read and write of every batch)
   while the fused pass transposes and prefixes in a single traversal.  In
   situ the win is larger than the hot-cache microbenchmark suggests, because
   the full-line stores also remove the read-for-ownership traffic.
7. an AVX-512 path for the interpolation group kernel (four live 16-lane
   accumulators; same per-element FMA order as the 8-lane path), an AVX-512
   de-interleave of the amplitude window (`vpermt2ps` instead of 32 scalar
   moves per 16 complex) and a 16-wide complex rotate for fully interior
   groups.  Interpolation: 0.42 s -> 0.38 s.
8. the interpolation table sizes were measured while evaluating this step:
   `gW` (the per-group expanded weights) is 1.71 MB in total and the compact
   per-harmonic `W[p][i]` table 0.19 MB, with the group count `ngrp` between 1
   and 15 per harmonic (`ngrp = q/gcd(V*s, q)`, not `q`).
   That answers the obvious question: replacing `gW` by the compact table
   would need a 32-way gather per tap, and a `vgatherdps` pair costs ~5-12
   cycles where the two contiguous 64-byte loads cost ~1, i.e. an order of
   magnitude worse for a 1.5 MB saving that already fits L3.  Measured and
   rejected.

Ablating the interpolation pass (with temporary instrumentation) splits its
0.416 s as
FMA accumulation 0.157, complex rotate + scattered stores 0.139, amplitude
window copy 0.039, plan bookkeeping and zero padding 0.034, with the rest in
interactions.  The stores are the part that is not instruction-bound: each
group writes 32 eight-byte complex values at a 488-byte stride, so a trial's
61 columns need eight separate harmonic passes and every 8-byte store re-reads
its cache line for ownership.  The fix is a harmonic-block x trial-group tile
(eight harmonics fill exactly one 64-byte line of a trial) so each line is
written once, but the row length of `nharms+1` complex makes the block starts
shift by 40 bytes per trial, so the flush must be done with unaligned 64-byte
stores; the estimated gain is ~0.1 s and the restructuring is invasive, so it
is left as the next step rather than done here.

End to end, the 105-file DM list at `--threads 1` went from 215.3 s to 146.7 s
(-32%), and a single 305 s observation from 2.06 s to 1.33 s (-35%), with
byte-identical `.cohout` files on all 105 inputs; at 8 threads the same file
went from 0.29 s to 0.22 s.  Phase totals for that observation are now
interp 0.38, base FFT 0.23, base metric 0.11, decimated FFTs 0.34, decimated
metrics 0.27 s (1.90 s before), and the boxcar gate is transpose 0.15 +
scan 0.19 s.  A `--blocksize` sweep from 256 to 8192 on that observation gave
1.40-1.50 s with identical candidates, so the default stays at 2048.  `--precision f64`, `--sigma measured`,
`--blocksize 512/2048/8192` and `--maxdecim 4/7` were also diffed against the
previous binary and are unchanged.

The GPU path gained two changes:

- the per-chunk full memset of `d_ftprofs` (32 MB at the default GPU block
  size) was replaced by a kernel that zeroes only the columns the
  interpolation pass does not write (DC plus the harmonics that do not fit the
  chunk), matching `fill_chunk_profiles` on the CPU.  105-file list:
  5.72 s -> 5.63 s.
- `fold_metric_kernel` fuses the fold and the boxcar metric for every
  decimation with `k | nharms` (8 of the 15).  It builds the base prefix sums
  once in shared memory, derives each folded prefix with
  `pf[m] = (1/k) * sum_q (P[m+q*nk] - P[q*nk])` plus the cyclic tail
  `pf[nk+u] = pf[nk] + pf[u]`, and runs the same window scan as
  `metric_kernel`.  The decimated profile never exists in global memory, and
  those decimations create no cuFFT plan or `dprofs` buffer at all.  The
  decimated phase fell from 28.9 ms to 26.9 ms per 305 s observation and the
  105-file list from 5.63 s to 5.39 s (5.35 s after the CPU boxcar change),
  with byte-identical `.cohout` files;
  the smaller startup (7 cuFFT plans instead of 15) is what shows up most on
  single-file runs.  The kernel needs f32 and ~24 KB of shared memory per
  block; with `--precision f64`, or when the shared-memory budget is short,
  every decimation falls back to cuFFT + `metric_kernel`.  Two variants were
  measured and rejected: dropping the shared base prefix and folding from L1
  was 2.8x slower on the decimated phase (75.0 ms), and folding only
  `k <= 8` was slower than folding all divisors (27.6 ms vs 26.9 ms).

Where each optimisation transfers, checked one by one:

- **exact fold.**  Only the decimated transforms subsample a spectrum, so there
  is nothing else to fold.  `candidate_profile` rebuilds one profile by direct
  interpolation, not an FFT, and the base transform has no sub-spectrum.
- **scan loop interchange.**  Transfers to the CPU gate (done).  It does *not*
  transfer to the GPU `metric_kernel`/`fold_metric_kernel`: with the
  accumulators unrolled to a compile-time bound, the predicated-off iterations
  still consume issue slots, and at nw = 4..9 with the 3-4 warps/SM that the
  shared prefix allows, the base metric went from 4.5 ms to 11.4 ms and the
  decimated phase from 26.9 ms to 40.0 ms.  Reverted; the GPU keeps
  width-outer.
- **bit-exact AVX-512.**  Done for the interpolation FMA loop, the amplitude
  window de-interleave and the complex rotate.  The remaining candidates are
  small or non-default: `block_sigma`'s copy+`fabs` pass (the `--sigma
  measured` mode costs +0.4 s on a 1.45 s run, but that is the exact
  quickselect median, not the copy), the `profile_boxcar_f32` tail/re-score
  path (no tail at the default block size; re-scores only for candidates above
  `exactcut`), the non-rational `--hidr` fallback, and the `datpipe`
  power/normalise loops (33 ms/file: 20 ms FFTW `ESTIMATE` realfft, which is a
  deliberate PRESTO-port choice, and 13 ms O(n) rednoise; in list mode both are
  hidden by the two-producer prefetch pipeline).
- **full-cache-line stores.**  `psT` was the only stream with partial-line
  writes.
- **cache blocking of the CPU gate** was measured: `CS_BC_BATCH` at 16, 32,
  64, 128 and 256 gave 1.34-1.40 s on the heavy observation, i.e. no effect.
  The base metric's `psT` (76 KB at B=128) already sits in L2, and the scan is
  bound by the number of vector loads (nbins*nw per 16 profiles), not by
  capacity, so shrinking the working set buys nothing.
- **fusing the gate's prefix and scan** (replace `psT` with a wmax-deep ring
  buffer) was analysed and, on the CPU, left undone for the same reason: a ring
  moves the loads from L2 to L1 but does not reduce their count.
- the same idea was implemented for the **GPU** `metric_kernel` (shared
  19 KB -> 4 KB, occupancy ~4 -> ~25 CTAs/SM, scan fused into the prefix pass,
  widths template-unrolled so there is no predication).  It is correct on all
  105 files but **6% slower** (105-file list 5.37 s -> 5.72 s) and was
  reverted: the width-outer scan walks two contiguous shared rows per width,
  while the fused position-outer pass interleaves wmax+1 strided row accesses
  per position.  This matches the earlier loop-interchange result, so the GPU
  metric keeps width-outer.
- **fusing the non-divisor decimated transforms on the GPU** (direct DFT of
  the k=7,8,9,11,13,14,16 sub-spectra inside the metric kernel, as done for
  the foldable ones) is the next GPU step: one kernel for all seven would cut
  the decimated traffic from ~57 MB to ~33 MB per chunk (~5 ms per
  observation).  A kernel per k would be *worse*, because each would re-read
  the 488-byte trial row and the 32-byte sector granularity dominates for
  Hk+1 <= 5 harmonics.
- **removing redundant traffic.**  `fill_chunk_profiles` already zeroes only
  the columns it does not write, the `.fft` reader is mmap-based, and applying
  a zaplist costs nothing measurable (1.50 s with and without `--zaplist`).
- **post-processing.**  Collapsing 1298 above-threshold candidates is not
  measurable either (1.43-1.50 s with and without `--noremove
  --noharmremove`).

After these changes the remaining GPU kernels are at the memory-bandwidth
limit (about 32 GB of traffic per 305 s observation), so further gains there
need to remove passes, not reorder them.  All timing/profiling switches used
during this work have been removed from the shipped build; the numbers above
are from temporary instrumentation.

### The prefetch pipeline

(The former `-M/--batch` interleaved-batch drivers `coherent_search_list_m`
and `prep_coherent_search_m` were removed: on the GPU they fell back to the
one-file path anyway, and on the CPU they were slower at the default block
size (four concurrent per-file working sets overflow the L2/L3 locality that
the single-file driver enjoys), so they no longer justified the extra code.
Their outputs were always byte-identical to the plain drivers'.)

A few background producer threads read and prepare upcoming files (the
in-memory realfft + rednoise for `.dat` inputs) while the main thread searches
the current one.  With two producers and four slots the ~0.05 s/file
preparation is fully hidden behind even a GPU search.  The pipeline runs
whenever more than one input is searched; a process-wide FFTW planner mutex
keeps the producers' plan creation safe against the search thread (plan
execution remains parallel).

### Compiler tuning

The default build already uses `-O3 -march=native -funroll-loops` with strict
`-ffp-contract=off`.  `LTO=1` adds GCC link-time optimization
(`-flto=auto -fno-semantic-interposition`), allowing cross-file inlining and
constant propagation without changing floating-point semantics.  On the
105-file CPU list it measured 22.6-23.5 s (median ~22.9 s) versus 23.2-23.3 s
for the default build, with byte-identical candidates; the heavy single-thread
case improved from ~8.3 s to ~8.2 s.  The gain is small because most of the hot
work is already in explicit AVX2/AVX-512 kernels and in FFTW/cuFFT.

Other compiler experiments were not beneficial and are not enabled:

- `-O2` was ~10-15% slower than `-O3` on the heavy case;
- PGO (`-fprofile-generate` / `-fprofile-use`) was slightly slower
  single-threaded and neutral with 8 threads;
- on this Zen 4/5 host, `-mprefer-vector-width=256/512`,
  `-fno-stack-protector`, `-fno-math-errno/-fno-trapping-math`, and
  `-funroll-all-loops` gave no consistent gain;
- `-Ofast` / `-ffast-math` are deliberately excluded: they allow reassociation
  and contraction and therefore change the last bits of S/N values.

`VECTOR_WIDTH=native` (the default) leaves the choice to `-march=native`, which
currently prefers 128-bit vectors on Zen 4/5.  `VECTOR_WIDTH=auto` inspects
`/proc/cpuinfo`: 256 on an AMD CPU with AVX-512, 512 on an Intel CPU with
AVX-512, and native otherwise.  You can also force `128`, `256`, or `512` with
`make VECTOR_WIDTH=...`; the option only changes GCC's vector-width preference,
not the ISA or numerical results.  On this host the forced widths were within
noise of native, but a native AVX-512 server may benefit from `auto` or `512`.

Only GCC 13 was available on the benchmark host; Clang/AOCC were not installed,
so their code generation was not tested.  Vendor libraries such as AOCL gain
most of their speed from hand-tuned assembly and PGO-tuned kernels, which is a
different lever from compiler flags on these already-specialised kernels.

For the CUDA side, `-arch=native` already produced `sm_89` SASS for the
RTX 4080 SUPER.  `CUDA_ARCH=-arch=sm_89` (or the appropriate `sm_XX`) can be
used when cross-building, and `NVCC_EXTRA` passes extra `nvcc`/`ptxas` flags.
The tested flags (`-Xptxas -dlcm=ca/cg`, `--extra-device-vectorization`,
`--allow-expensive-optimizations=true`) gave no measurable gain, and ptxas
reported no spills.  The useful GPU win came from removing the decimation
gather kernel: cuFFT now reads `d_ftprofs` with `istride=k` and
`idist=nharms+1` directly, cutting the 105-file list from about 7.0 s to
5.7 s.  The metric kernel remains occupancy-limited by its per-warp shared
prefix sums (three blocks/SM here); block-size and carveout experiments did
not improve it.

## Known limitations

- `--gpu` requires a `GPU=1` build.  With `GPU=0`, or on a machine with no
  accessible CUDA device, it warns and runs the CPU path.
- `prep_coherent_search` builds in both configurations.  In a `GPU=0` build
  the dedispersion is CPU-only (the `-cuda` option is rejected) and no CUDA or
  external PRESTO library is linked; `GPU=1` additionally supports `-cuda`.
  It keeps all DMs of one prep command in RAM (about 0.7-1.4 GB for the
  provided 210/50-DM examples).  `-sub` is rejected and
  `-outlist` is ignored because no `.dat` files exist.
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
                            AVX2/AVX-512 batched gate + fused 16-wide prefix
                            transpose
src/search.c                workspaces, FFTW plans, decimation, threaded chunk
                            loop, metric statistics, normalisation, wisdom
src/candidate.c             candidate lists/collapsing, profile reconstruction,
                            duty cycle, output formatting
src/datpipe.c               in-memory realfft + rednoise (.dat input)
src/prep_dedisp.c           prepsubband dedispersion, memstream output + .inf
src/prep_search.c           in-memory FFTFile -> realfft/rednoise -> search
src/prep_cli.c              prep/search CLI split at ++
src/prep_main.c             main() for prep_coherent_search
include/prep_search.h       shared prep/search options
third_party/presto/         vendored PRESTO headers + prepsubband sources
src/cli.c                   argument parsing, list mode, --metricstats
src/plot.c                  dependency-free SVG candidate-profile pages
src/cuda.cu                 CUDA backend (interp, cuFFT, metric)
src/gpu_stub.c              no-CUDA stubs for GPU=0 builds
include/cuda.cuh            CUDA-internal declarations
src/CoherentSearch.c        main() for coherent_search
src/list_main.c             main() for coherent_search_list
python/sift_candidates.py   cross-observation/DM candidate sifting (no Julia)
python/parallel_search.py   split a file list over CPU jobs or CUDA devices
bin/sift_candidates.py      symlink to python/sift_candidates.py
bin/parallel_search.py      symlink to python/parallel_search.py
bin/prep_coherent_search    raw-data prep + coherent search
src/Makefile                build (GPU=0/1)
```
