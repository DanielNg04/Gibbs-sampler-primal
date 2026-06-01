# sdp_conversion.py
#
# Convert SDPLIB-style problems given in the SDPA *sparse* format (``.dat-s``)
# into the input form required by the quantum primal oracle
# (``primal_oracle_quantum_v1cube.run_primal_oracle``).

"""
Convert SDPA sparse (``.dat-s``) files to the primal-oracle form used by ``run_test.py``.

Dual ``(D): max tr(F0 Y) s.t. tr(Fi Y)=ci, Y⪰0`` becomes inequalities on ``Y`` plus trace bound
``Tr(Y) ≤ R``. Default ``R = 1.1·Tr(Y*)`` from a CVXPY dual solve; ``--R`` overrides.

Output: ``<output_dir>/<name>.npz`` and ``manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Iterable, Literal

RSource = Literal["override", "cvxpy", "heuristic"]
DEFAULT_R_MARGIN_SOLVER = 1.1
DEFAULT_R_MARGIN_HEURISTIC = 2.0

import numpy as np

DTYPE = np.float64

# Punctuation that the SDPA spec allows inside the block-size / objective lines.
_PUNCT_RE = re.compile(r"[(){},]")
_LEADING_ALPHA_RE = re.compile(r"^[A-Za-z]+")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@dataclass
class SDPAProblem:
    """Raw, dense SDPA problem assembled from a ``.dat-s`` file."""

    name: str
    m: int                       # number of constraint matrices F_1..F_m
    block_sizes: list[int]       # signed block sizes (negative = diagonal block)
    n: int                       # total matrix dimension Σ |block_size|
    c: np.ndarray                # objective vector (length m)
    F: list[np.ndarray]          # F[0..m], each n×n symmetric


def _tokenize_after_comments(text: str) -> list[str]:
    """
    Drop leading comment lines (each starting with ``"`` or ``*``) and return all
    remaining whitespace-separated tokens, with SDPA punctuation removed.

    Tokenizing the whole remainder (rather than parsing line-by-line) is robust to
    block-size or objective vectors that wrap across lines.
    """
    kept_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[0] in ('"', "*"):
            continue
        kept_lines.append(line)
    joined = " ".join(kept_lines)
    joined = _PUNCT_RE.sub(" ", joined)
    return joined.split()


def parse_sdpa_sparse(path: str) -> SDPAProblem:
    """Parse a single SDPA-sparse ``.dat-s`` file into dense symmetric matrices."""
    name = os.path.splitext(os.path.basename(path))[0]
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        tokens = _tokenize_after_comments(fh.read())

    if len(tokens) < 2:
        raise ValueError(f"{name}: file too short / unparsable.")

    pos = 0
    m = int(float(tokens[pos])); pos += 1
    nblocks = int(float(tokens[pos])); pos += 1

    block_sizes = [int(float(tokens[pos + k])) for k in range(nblocks)]
    pos += nblocks

    c = np.array([float(tokens[pos + k]) for k in range(m)], dtype=DTYPE)
    pos += m

    # Block offsets within the flattened n×n matrix.
    abs_sizes = [abs(b) for b in block_sizes]
    n = int(sum(abs_sizes))
    offsets: list[int] = []
    running = 0
    for s in abs_sizes:
        offsets.append(running)
        running += s

    F = [np.zeros((n, n), dtype=DTYPE) for _ in range(m + 1)]

    entry_tokens = tokens[pos:]
    if len(entry_tokens) % 5 != 0:
        raise ValueError(
            f"{name}: trailing entry tokens not a multiple of 5 "
            f"(got {len(entry_tokens)})."
        )
    for k in range(0, len(entry_tokens), 5):
        matno = int(float(entry_tokens[k]))
        blkno = int(float(entry_tokens[k + 1]))
        i = int(float(entry_tokens[k + 2]))
        j = int(float(entry_tokens[k + 3]))
        val = float(entry_tokens[k + 4])
        if not (0 <= matno <= m):
            raise ValueError(f"{name}: matno {matno} out of range [0, {m}].")
        if not (1 <= blkno <= nblocks):
            raise ValueError(f"{name}: blkno {blkno} out of range [1, {nblocks}].")
        off = offsets[blkno - 1]
        gi = off + i - 1
        gj = off + j - 1
        F[matno][gi, gj] += val
        if gi != gj:
            F[matno][gj, gi] += val  # only upper triangle is stored; symmetrize.

    return SDPAProblem(name=name, m=m, block_sizes=block_sizes, n=n, c=c, F=F)


# ---------------------------------------------------------------------------
# Reduction to oracle form
# ---------------------------------------------------------------------------


@dataclass
class OracleInstance:
    """Converted instance ready for the primal oracle."""

    name: str
    C: np.ndarray                # objective matrix F_0 (n×n)
    A: np.ndarray                # constraint stack (M, n, n): [I, F_1, -F_1, ...]
    b: np.ndarray                # bounds (M,): [R, c_1, -c_1, ...]
    R: float                     # trace bound (b_1)
    opt: float | None            # known optimum (objective value), if available
    n: int = field(init=False)
    m_oracle: int = field(init=False)

    def __post_init__(self) -> None:
        self.n = int(self.C.shape[0])
        self.m_oracle = int(self.b.shape[0])


def trace_bound_from_opt(opt: float | None, *, margin: float) -> float:
    """Legacy fallback: ``R = margin · max(|OPT|, 1)`` when the CVXPY solve fails."""
    base = 1.0 if opt is None else max(abs(float(opt)), 1.0)
    return float(margin) * base


@dataclass
class DualSolveResult:
    """Result of solving the SDPA dual to set ``R`` from ``Tr(Y*)``."""

    opt_value: float
    trace_y: float
    R: float
    status: str
    solver: str


def solve_sdpa_dual_for_trace_bound(
    problem: SDPAProblem,
    *,
    margin: float,
    solver: str | None = None,
    verbose: bool = False,
) -> DualSolveResult | None:
    """
    Solve the SDPA dual (D) and return ``R = margin · Tr(Y*)``.

    Dual program (same as the oracle reduction without the trace inequality)::

        max   tr(F_0 Y)
        s.t.  tr(F_i Y) = c_i   (i = 1..m)
              Y ⪰ 0

    Strong duality gives ``opt = tr(F_0 Y*)``. The returned ``Y`` is symmetrized and
    ``Tr(Y*)`` is used for the trace bound so the oracle ball contains the optimizer.
    """
    try:
        import cvxpy as cp
    except ImportError:
        if verbose:
            print(f"[cvxpy] {problem.name}: cvxpy not installed; skipping dual solve.")
        return None

    n = problem.n
    F0 = np.ascontiguousarray(problem.F[0], dtype=DTYPE)
    Y = cp.Variable((n, n), symmetric=True)
    objective = cp.Maximize(cp.trace(F0 @ Y))
    constraints: list = [Y >> 0]
    for i in range(1, problem.m + 1):
        Fi = np.ascontiguousarray(problem.F[i], dtype=DTYPE)
        constraints.append(cp.trace(Fi @ Y) == float(problem.c[i - 1]))

    prob = cp.Problem(objective, constraints)
    solver_chain: list[str | None] = []
    if solver:
        solver_chain.append(solver)
    solver_chain.extend(["CLARABEL", "SCS", "ECOS"])

    last_status = "not_attempted"
    for sol in solver_chain:
        if sol is None:
            continue
        try:
            prob.solve(solver=sol, verbose=verbose)
        except Exception as exc:
            if verbose:
                print(f"[cvxpy] {problem.name}: solver {sol!r} failed: {exc}")
            continue
        last_status = str(prob.status)
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            continue
        Y_val = Y.value
        if Y_val is None:
            continue
        Y_sym = np.asarray((Y_val + Y_val.T) / 2.0, dtype=DTYPE)
        trace_y = float(np.trace(Y_sym).real)
        if not np.isfinite(trace_y) or trace_y <= 0.0:
            if verbose:
                print(f"[cvxpy] {problem.name}: non-positive Tr(Y*)={trace_y}")
            continue
        opt_val = float(prob.value)
        return DualSolveResult(
            opt_value=opt_val,
            trace_y=trace_y,
            R=float(margin) * trace_y,
            status=last_status,
            solver=str(sol),
        )

    if verbose:
        print(f"[cvxpy] {problem.name}: no solver succeeded (last status {last_status!r}).")
    return None


def resolve_trace_bound(
    problem: SDPAProblem,
    *,
    opt_csv: float | None,
    R_margin: float,
    heuristic_margin: float,
    R_override: float | None,
    R_from_solver: bool,
    solver: str | None = None,
    verbose: bool = False,
) -> tuple[float, RSource, float | None, float | None]:
    """
    Choose ``R`` and optional solver-derived ``opt`` / ``Tr(Y*)``.

    Returns ``(R, source, tr_Y_star, opt_solver)``.
    """
    if R_override is not None:
        return float(R_override), "override", None, None

    if R_from_solver:
        solved = solve_sdpa_dual_for_trace_bound(
            problem,
            margin=R_margin,
            solver=solver,
            verbose=verbose,
        )
        if solved is not None:
            return solved.R, "cvxpy", solved.trace_y, solved.opt_value

    R = trace_bound_from_opt(opt_csv, margin=heuristic_margin)
    return R, "heuristic", None, None


def reduce_to_oracle(
    problem: SDPAProblem,
    *,
    R: float,
    opt: float | None,
) -> OracleInstance:
    """Apply the dual reduction (equality → ± inequalities, prepend trace bound)."""
    n = problem.n
    C = np.ascontiguousarray(problem.F[0], dtype=DTYPE)

    A_list: list[np.ndarray] = [np.eye(n, dtype=DTYPE)]
    b_list: list[float] = [float(R)]
    for i in range(1, problem.m + 1):
        Fi = problem.F[i]
        A_list.append(Fi)
        b_list.append(float(problem.c[i - 1]))
        A_list.append(-Fi)
        b_list.append(float(-problem.c[i - 1]))

    A = np.ascontiguousarray(np.stack(A_list, axis=0), dtype=DTYPE)
    b = np.asarray(b_list, dtype=DTYPE)
    return OracleInstance(name=problem.name, C=C, A=A, b=b, R=float(R), opt=opt)


# ---------------------------------------------------------------------------
# OPT-value table and class names
# ---------------------------------------------------------------------------


_OPT_VALUE_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def load_opt_values(csv_path: str) -> dict[str, float]:
    """
    Read the OPT table into ``{instance_name: opt}``.

    The file is only loosely CSV-shaped and mixes several decorations per line:
    some rows are clean (``ThetaPrimeER23_red, -96.240038;``) while others wrap
    the whole ``name, value`` pair in double quotes and append stray separators
    or tabs (``"control1, 1.778463e+01\t";,``). To be robust we:

    1. strip a leading wrapping quote and split on the *first* comma to get the
       instance name (everything after the comma is the value field), then
    2. pull the value out of the (possibly tab/quote/semicolon-littered) field
       with a numeric regex rather than ``float()`` on the raw text.
    """
    out: dict[str, float] = {}
    with open(csv_path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip().lstrip('"').strip()
            if not line or "," not in line:
                continue
            name, _, val = line.partition(",")
            name = name.strip().strip('"').strip()
            if name.lower() == "instance":
                continue  # header row
            match = _OPT_VALUE_RE.search(val)
            if match is None:
                continue
            try:
                out[name] = float(match.group(0))
            except ValueError:
                continue
    return out


def instance_class(name: str) -> str:
    """Problem class = leading alphabetic prefix (e.g. ``ThetaPrimeER23_red`` → ``ThetaPrimeER``)."""
    match = _LEADING_ALPHA_RE.match(name)
    return match.group(0) if match else name


# ---------------------------------------------------------------------------
# Conversion driver
# ---------------------------------------------------------------------------


def convert_instance(
    dat_path: str,
    *,
    opt_csv: float | None,
    R_margin: float,
    heuristic_margin: float,
    R_override: float | None,
    R_from_solver: bool,
    cvxpy_solver: str | None,
    verbose_cvxpy: bool,
    out_dir: str,
) -> dict:
    """Parse, reduce, and write one instance; return its manifest entry."""
    problem = parse_sdpa_sparse(dat_path)
    R, R_source, tr_Y_star, opt_solver = resolve_trace_bound(
        problem,
        opt_csv=opt_csv,
        R_margin=R_margin,
        heuristic_margin=heuristic_margin,
        R_override=R_override,
        R_from_solver=R_from_solver,
        solver=cvxpy_solver,
        verbose=verbose_cvxpy,
    )
    opt = opt_solver if opt_solver is not None else opt_csv
    inst = reduce_to_oracle(problem, R=R, opt=opt)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{inst.name}.npz")
    np.savez_compressed(
        out_path,
        C=inst.C,
        A=inst.A,
        b=inst.b,
        R=np.float64(inst.R),
        n=np.int64(inst.n),
        m_oracle=np.int64(inst.m_oracle),
        opt=np.float64(np.nan if inst.opt is None else inst.opt),
        opt_csv=np.float64(np.nan if opt_csv is None else opt_csv),
        tr_Y_star=np.float64(np.nan if tr_Y_star is None else tr_Y_star),
        R_source=np.array(R_source),
        name=inst.name,
    )

    entry = {
        "name": inst.name,
        "class": instance_class(inst.name),
        "n": inst.n,
        "m_oracle": inst.m_oracle,
        "R": inst.R,
        "R_source": R_source,
        "tr_Y_star": None if tr_Y_star is None else float(tr_Y_star),
        "opt": None if inst.opt is None else float(inst.opt),
        "opt_csv": None if opt_csv is None else float(opt_csv),
        "npz": os.path.basename(out_path),
    }
    return entry


def _select_dat_files(input_dir: str, instances: Iterable[str] | None) -> list[str]:
    """List ``.dat-s`` files in ``input_dir`` filtered by optional name substrings."""
    files = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".dat-s")
    )
    if instances:
        wanted = [s.lower() for s in instances]
        files = [f for f in files if any(w in os.path.basename(f).lower() for w in wanted)]
    return files


def convert_directory(
    input_dir: str,
    output_dir: str,
    *,
    opt_csv: str | None,
    R_margin: float,
    heuristic_margin: float,
    R_override: float | None,
    R_from_solver: bool,
    cvxpy_solver: str | None,
    verbose_cvxpy: bool,
    instances: Iterable[str] | None,
) -> list[dict]:
    """Convert every selected instance and write ``manifest.json``."""
    opt_values = load_opt_values(opt_csv) if opt_csv and os.path.exists(opt_csv) else {}
    dat_files = _select_dat_files(input_dir, instances)
    if not dat_files:
        raise FileNotFoundError(f"No matching .dat-s files in {input_dir!r}.")

    manifest: list[dict] = []
    for path in dat_files:
        name = os.path.splitext(os.path.basename(path))[0]
        opt = opt_values.get(name)
        entry = convert_instance(
            path,
            opt_csv=opt,
            R_margin=R_margin,
            heuristic_margin=heuristic_margin,
            R_override=R_override,
            R_from_solver=R_from_solver,
            cvxpy_solver=cvxpy_solver,
            verbose_cvxpy=verbose_cvxpy,
            out_dir=output_dir,
        )
        manifest.append(entry)
        tr_txt = (
            f" Tr*={entry['tr_Y_star']:.4g}"
            if entry.get("tr_Y_star") is not None
            else ""
        )
        print(
            f"[convert] {entry['name']:<24} class={entry['class']:<12} "
            f"n={entry['n']:<5} M={entry['m_oracle']:<6} "
            f"R={entry['R']:.4g} ({entry['R_source']}){tr_txt} "
            f"opt={entry['opt']}"
        )

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[convert] wrote {len(manifest)} instances + manifest to {output_dir}")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Convert SDPA-sparse problems to primal-oracle .npz instances.",
    )
    parser.add_argument(
        "--input-dir",
        default=os.path.join(here, "SDP_problems"),
        help="Directory containing the .dat-s files.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(here, "SDP_problems_converted"),
        help="Directory to write the converted .npz files and manifest.json.",
    )
    parser.add_argument(
        "--opt-csv",
        default=os.path.join(here, "SDP_problems", "Meta", "Instance_-OPT_value.csv"),
        help="CSV with 'Instance, OPT_value' rows (used to derive R and as the g reference).",
    )
    parser.add_argument(
        "--R-margin",
        type=float,
        default=DEFAULT_R_MARGIN_SOLVER,
        help="With --R-from-solver: R = R_margin * Tr(Y*) from the CVXPY dual solution.",
    )
    parser.add_argument(
        "--heuristic-R-margin",
        type=float,
        default=DEFAULT_R_MARGIN_HEURISTIC,
        help="Fallback when the solver fails: R = margin * max(|OPT|, 1) from CSV.",
    )
    parser.add_argument(
        "--R-from-solver",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Solve the SDPA dual with CVXPY to set R from Tr(Y*) (default: on).",
    )
    parser.add_argument(
        "--cvxpy-solver",
        default=None,
        help="CVXPY solver name (default: try CLARABEL, then SCS, then ECOS).",
    )
    parser.add_argument(
        "--verbose-cvxpy",
        action="store_true",
        help="Print CVXPY solver logs and failure diagnostics.",
    )
    parser.add_argument(
        "--R",
        dest="R_override",
        type=float,
        default=None,
        help="Override the trace bound R for ALL converted instances.",
    )
    parser.add_argument(
        "--instances",
        nargs="*",
        default=None,
        help="Optional name substrings to filter which instances to convert.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    convert_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        opt_csv=args.opt_csv,
        R_margin=args.R_margin,
        heuristic_margin=args.heuristic_R_margin,
        R_override=args.R_override,
        R_from_solver=args.R_from_solver,
        cvxpy_solver=args.cvxpy_solver,
        verbose_cvxpy=args.verbose_cvxpy,
        instances=args.instances,
    )


if __name__ == "__main__":
    main()
