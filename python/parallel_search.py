#!/usr/bin/env python3
"""Run one coherent search over many .fft/.dat files as N independent jobs.

    python/parallel_search.py [-j N] [--gpu] [OPTIONS] FILE ... [-- SEARCH ARGS]

The C search binary parallelises *inside* one process (OpenMP on the CPU,
one CUDA device on the GPU).  To use a whole machine with several independent
processes, split the file list over disjoint jobs.  This script does that, with
the same behaviour as the Julia version it replaces:

  * **One invocation per job, not one per file.**  Start-up planning, the
    realfft/rednoise workspace, the search workspaces and (on the GPU) the CUDA
    context/cuFFT plans are amortised over each job's slice.
  * **A one-file job without --outdir would write to stdout.**  When that can
    happen, this script supplies the input's directory as --outdir so every
    file gets its `<stem>.cohout` exactly as a multi-file run would.

Files are dealt to the least-loaded job by size (largest first); --split block
keeps each job's slice contiguous.

Examples
--------
  # 6 GPUs, one job per card
  python/parallel_search.py --gpu -j 6 *.fft -- --blocksize 65536

  # a whole CPU socket, one single-threaded process per core
  python/parallel_search.py -j 20 -t 1 *.fft -- --threshold 8

  # resume an interrupted run
  python/parallel_search.py --gpu -j 6 --skip-existing --outdir cands *.fft

  # raw .dat time series (FFT + rednoise are done inside each job)
  python/parallel_search.py -j 8 -t 1 dm_*.dat -- --maxdecim 16
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SEARCH = REPO / "bin" / "coherent_search"


def physical_cores():
    """Physical core count (unique package/core_id), respecting CPU affinity."""
    pairs = set()
    try:
        allowed = os.sched_getaffinity(0)
    except AttributeError:
        allowed = None
    for cpu in range(os.cpu_count() or 1):
        if allowed is not None and cpu not in allowed:
            continue
        base = f"/sys/devices/system/cpu/cpu{cpu}/topology"
        try:
            with open(os.path.join(base, "core_id")) as fh:
                core = fh.read().strip()
            with open(os.path.join(base, "physical_package_id")) as fh:
                pkg = fh.read().strip()
            pairs.add((pkg, core))
        except OSError:
            pass
    return len(pairs) or (os.cpu_count() or 1)


def visible_gpus():
    """Device ordinals this process may use, honouring CUDA_VISIBLE_DEVICES."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None and env.strip() != "":
        return [d.strip() for d in env.split(",") if d.strip() != ""]
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return []
    try:
        out = subprocess.run([smi, "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        return [l.strip() for l in out.stdout.splitlines() if l.strip()]
    except Exception:
        return []


def partition(files, njobs, mode):
    """Split `files` into `njobs` lists.  Never returns an empty job."""
    njobs = max(1, min(njobs, len(files)))
    if mode == "block":
        out, start = [], 0
        for j in range(njobs):
            n = len(files) // njobs + (1 if j < len(files) % njobs else 0)
            out.append(files[start:start + n])
            start += n
        return out
    sized = sorted(files, key=lambda f: -Path(f).stat().st_size)
    out = [[] for _ in range(njobs)]
    load = [0] * njobs
    for f in sized:
        j = load.index(min(load))
        out[j].append(f)
        load[j] += Path(f).stat().st_size
    return [sorted(g) for g in out]


def cohout_for(path, outdir):
    """Candidate file the search will write for `path`."""
    p = Path(path)
    name = p.with_suffix("").name
    if outdir:
        return Path(outdir) / (name + ".cohout")
    return p.parent / (name + ".cohout")


def main():
    p = argparse.ArgumentParser(
        description="Run coherent_search over a file list as independent jobs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Anything after `--` is passed through to bin/coherent_search unchanged.")
    p.add_argument("files", nargs="+", help=".fft/.dat files (shell globs are fine)")
    p.add_argument("-j", "--jobs", type=int, default=0,
                   help="concurrent jobs; default = one per GPU with --gpu, "
                        "else the physical core count")
    p.add_argument("-t", "--threads", type=int, default=1,
                   help="OpenMP threads PER JOB (default 1; -j jobs x -t threads "
                        "should not exceed your cores)")
    p.add_argument("--gpu", action="store_true",
                   help="run each job on its own CUDA device via "
                        "CUDA_VISIBLE_DEVICES and pass --gpu 0 to the search")
    p.add_argument("--gpus", default="",
                   help="comma-separated device ordinals to use (default: all visible)")
    p.add_argument("--outdir", default="",
                   help="passed to the search; also where --skip-existing looks")
    p.add_argument("--split", choices=("size", "block"), default="size",
                   help="size (default): balance total bytes per job.  block: "
                        "contiguous slices")
    p.add_argument("--skip-existing", action="store_true",
                   help="drop inputs whose .cohout already exists (resume a partial run)")
    p.add_argument("--search", default=str(SEARCH),
                   help="coherent_search executable (default: bin/coherent_search)")
    p.add_argument("--logdir", default="",
                   help="per-job logs (default: a temp dir, path printed)")
    p.add_argument("-n", "--dry-run", action="store_true",
                   help="print the commands and exit")
    # Split on the first bare `--` ourselves (argparse would otherwise swallow it).
    argv = sys.argv[1:]
    if "--" in argv:
        cut = argv.index("--")
        argv, passthrough = argv[:cut], argv[cut + 1:]
    else:
        passthrough = []
    args = p.parse_args(argv)

    search = Path(args.search)
    if not search.is_file() and not args.dry_run:
        sys.exit("search executable not found: %s" % search)

    files = [f for f in args.files if Path(f).is_file()]
    missing = sorted(set(args.files) - set(files))
    if missing:
        print(f"warning: skipping {len(missing)} path(s) that are not files, "
              f"e.g. {missing[0]}", file=sys.stderr)
    if not files:
        sys.exit("no input .fft/.dat files")

    if args.skip_existing:
        before = len(files)
        files = [f for f in files if not cohout_for(f, args.outdir).exists()]
        print(f"--skip-existing: {before - len(files)} of {before} already done, "
              f"{len(files)} to run")
        if not files:
            return 0

    devices = []
    if args.gpu:
        devices = ([d.strip() for d in args.gpus.split(",") if d.strip()]
                   if args.gpus else visible_gpus())
        if not devices:
            sys.exit("--gpu given but no CUDA devices found (try --gpus 0)")

    njobs = args.jobs or (len(devices) if args.gpu else physical_cores())
    groups = partition(files, njobs, args.split)
    njobs = len(groups)

    if args.outdir:
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
    logdir = Path(args.logdir) if args.logdir else Path(f".parallel_search.{os.getpid()}")
    logdir.mkdir(parents=True, exist_ok=True)

    procs, logs = [], []
    for j, group in enumerate(groups):
        cmd = [str(search)]
        if args.gpu:
            cmd += ["--gpu", "0"]
        outdir = args.outdir
        # A one-file job with no --outdir writes to stdout instead of a .cohout.
        if not outdir and len(group) == 1:
            outdir = str(Path(group[0]).parent or ".")
        if outdir:
            cmd += ["--outdir", outdir]
        if not any(a.startswith("--progress") or a.startswith("--noprogress")
                   for a in passthrough):
            cmd.append("--noprogress")
        cmd += passthrough + group

        dev = devices[j % len(devices)] if args.gpu else None
        if args.dry_run:
            pre = f"CUDA_VISIBLE_DEVICES={dev} OMP_NUM_THREADS={args.threads} " \
                if dev is not None else f"OMP_NUM_THREADS={args.threads} "
            print(f"# job {j}: {len(group)} files")
            print(pre + " ".join(cmd))
            continue

        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(args.threads)
        if dev is not None:
            env["CUDA_VISIBLE_DEVICES"] = dev
        log = logdir / f"job{j}.log"
        fh = open(log, "w")
        logs.append((log, fh))
        procs.append(subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env))

    if args.dry_run:
        return 0

    where = ("device " + ",".join(devices)) if args.gpu else \
            f"{args.threads} thread(s) each"
    print(f"{njobs} jobs over {len(files)} files ({where}); logs in {logdir}")
    t0 = time.time()
    tty = sys.stdout.isatty()
    pat = re.compile(r"Info: Searching")
    last = -1
    try:
        while any(pr.poll() is None for pr in procs):
            time.sleep(2.0)
            done = 0
            for log, fh in logs:
                fh.flush()
                try:
                    done += len(pat.findall(log.read_text(errors="replace")))
                except OSError:
                    pass
            if done != last:
                el = time.time() - t0
                msg = f"  {done}/{len(files)} started   {el:6.1f} s elapsed"
                print(("\r" + msg) if tty else msg,
                      end="" if tty else "\n", flush=True)
                last = done
    except KeyboardInterrupt:
        print("\ninterrupted; terminating jobs")
        for pr in procs:
            pr.terminate()
    codes = [pr.wait() for pr in procs]
    for _, fh in logs:
        fh.close()
    print(("\r" if tty else "") + f"done: {len(files)} files in "
          f"{time.time() - t0:.1f} s" + " " * 20)

    bad = [j for j, c in enumerate(codes) if c != 0]
    if bad:
        for j in bad:
            print(f"job {j} exited {codes[j]} -- see {logs[j][0]}", file=sys.stderr)
            print("  " + "\n  ".join((logs[j][0].read_text(errors="replace")
                                      .splitlines() or ["<empty log>"])[-8:]),
                  file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
