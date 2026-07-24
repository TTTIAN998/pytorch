"""Path-cached ``torch.compiler.export_python`` decorator.

``torch.compiler.precompile`` captures a function ahead of time and lowers it to a
self-contained, human-readable Python source artifact (see
``torch/_precompile.py``). ``torch.compiler.export_python`` wraps that in a
decorator keyed off a file on disk: the first run writes the emitted
``python_code`` to ``path``; every later run reads the ``.py`` back and executes
it directly instead of recompiling.

Because the artifact is self-contained, re-executable Python, ``path`` is meant to
be committed and hand-edited: an engineer or agent can "hill-climb" the generated
kernel in place. This is ejectable compilation -- the emitted source is the source
of truth and is always exec'd, so hand edits always take effect. There is no
acceleration cache and no ``precompile.load`` round-trip: the source is exec'd as
written, so keeping the edited source correct is the caller's responsibility.
"""

import copy
import functools
import inspect
import logging
import os
import tempfile
import threading
from collections.abc import Callable, Sequence
from typing import Any

import torch


log = logging.getLogger(__name__)

# Written as the artifact's first line so a later load can detect it was produced
# by a different torch (see _warn_on_version_skew). It is a comment, so it does not
# affect exec; a hand-edit that drops it just disables the skew warning, so
# hill-climbing an artifact never triggers a spurious version warning.
_VERSION_TAG = "# torch.compiler.export_python torch-version: "


def _atomic_write(path: str, data: bytes) -> None:
    # Write to a temp file in the same directory, fsync it, then rename it into
    # place. os.replace is atomic, so an interrupted or concurrent writer never
    # leaves a half-written artifact that the presence-only load gate would read
    # as valid.
    dir_name = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        # mkstemp creates the temp file 0600 and os.replace preserves that mode,
        # but this artifact is meant to be committed and hand-edited, so it needs
        # conventional world-readable perms rather than the private tempfile mode.
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


class ExportedPythonArtifact:
    """Materializes and disk-caches a ``torch.compiler.precompile`` artifact.

    Materialization is lazy and happens on the first call: if ``path`` exists the
    emitted Python is read from disk, otherwise the wrapped ``fn`` is precompiled
    against the example inputs and the emitted source is written to disk. Either
    way the source is exec'd directly to build the runnable. The loaded callable is
    reused for all subsequent calls in the process; a later process re-reads
    whatever is on disk.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        path: str,
        backend: str,
        tracer: str,
        decompositions: dict | None,
        example_inputs: Sequence[object] | None,
    ) -> None:
        self._fn = fn
        self._signature = inspect.signature(fn)
        self._path = path
        self._backend = backend
        self._tracer = tracer
        self._decompositions = decompositions
        self._example_inputs = None if example_inputs is None else tuple(example_inputs)
        self._loaded: Callable[..., Any] | None = None
        self._lock = threading.Lock()
        functools.update_wrapper(self, fn)

    def _precompile_and_save(self, args: tuple[Any, ...]) -> str:
        example = self._example_inputs
        if example is None:
            # Capture runs fn once on the example inputs (real-mode make_fx), which
            # mutates them; deep-copy the live call args so capture side effects (in-
            # place input mutation, module buffer updates) do not leak onto the
            # caller before the artifact itself runs on the real args exactly once.
            try:
                example = copy.deepcopy(args)
            except Exception as e:
                from torch._precompile import PrecompileError

                raise PrecompileError(
                    "torch.compiler.export_python could not deep-copy the "
                    "first-call arguments to capture without mutating them (e.g. a "
                    "non-leaf tensor or a weight_norm module). Pass explicit "
                    "example_inputs=... to precompile against dedicated inputs."
                ) from e
        # precompile returns (python_code, cache); the cache is an acceleration
        # artifact that export_python does not use -- the emitted source is
        # self-contained and always exec'd -- so only the code is written to disk.
        code, _cache = torch.compiler.precompile(
            self._fn,
            *example,
            backend=self._backend,
            tracer=self._tracer,
            decompositions=self._decompositions,
        )
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Stamp the producing torch version as the artifact's first line so a load
        # by a different torch can warn about the skew. It is a comment (exec-inert)
        # and a hand-edit may freely drop it (see _warn_on_version_skew).
        code = f"{_VERSION_TAG}{torch.__version__}\n{code}"
        # The write is atomic, so two first-call processes that both pass the
        # presence gate each write a complete, self-contained artifact and the last
        # rename wins; neither leaves a half-written file behind the gate.
        _atomic_write(self._path, code.encode("utf-8"))
        return code

    def _load_from_disk(self) -> str:
        with open(self._path, encoding="utf-8") as f:
            return f.read()

    def _warn_on_version_skew(self, code: str) -> None:
        # Warn (but still run) when the artifact carries a version stamp that does
        # not match the current torch, so a committed artifact gone stale across a
        # torch upgrade is visible rather than silently running old logic. A missing
        # stamp (dropped by a hand-edit) is silent, so hill-climbing never warns.
        first_line = code.split("\n", 1)[0]
        if not first_line.startswith(_VERSION_TAG):
            return
        produced = first_line[len(_VERSION_TAG) :].strip()
        if produced != torch.__version__:
            log.warning(
                "torch.compiler.export_python: the artifact at %s was produced by "
                "torch %s but the current torch is %s; running it as-is. Delete %s "
                "to regenerate against the current torch.",
                self._path,
                produced,
                torch.__version__,
                self._path,
            )

    def _load(self, code: str) -> Callable[..., Any]:
        # The emitted source is self-contained: exec it directly (no cache, no
        # precompile.load round-trip) and without the untrusted-exec warning, since
        # export_python only ever loads artifacts it produced. A clobbered hand-edit
        # (dropped forward / syntax error) and an environment or version mismatch (an
        # import that fails under the current torch) surface as distinct, actionable
        # PrecompileErrors rather than one catch-all "delete to regenerate".
        from torch._precompile import _make_inlined_forward, PrecompileError

        try:
            return _make_inlined_forward(code, warn=False)
        except (SyntaxError, KeyError) as e:
            raise PrecompileError(
                f"torch.compiler.export_python: the artifact at {self._path} could "
                "not be run as precompile source; it is not a valid "
                "torch.compiler.precompile artifact (a hand-edit may have clobbered "
                "it, e.g. dropping forward()). Delete it to regenerate."
            ) from e
        except ImportError as e:
            raise PrecompileError(
                f"torch.compiler.export_python: the artifact at {self._path} failed "
                "to import a dependency; it was likely produced by a different torch "
                f"version or environment. Delete {self._path} to regenerate against "
                "the current torch."
            ) from e
        except Exception as e:
            raise PrecompileError(
                "torch.compiler.export_python: an unexpected error occurred running "
                f"the artifact at {self._path}. Delete it to regenerate."
            ) from e

    def _materialize(self, args: tuple[Any, ...]) -> Callable[..., Any]:
        if os.path.exists(self._path):
            code = self._load_from_disk()
            self._warn_on_version_skew(code)
        else:
            code = self._precompile_and_save(args)
        return self._load(code)

    def _bind_positional(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[Any, ...]:
        # The artifact's forward is positional (the precompile calling convention),
        # so map any keyword call args onto fn's positional parameters -- this lets
        # callers invoke the decorated fn naturally (e.g. rope(q=..., k=...)).
        # Anything that cannot be laid out positionally is rejected below.
        sig = self._signature
        try:
            bound = sig.bind(*args, **kwargs)
        except TypeError as e:
            raise TypeError(
                "torch.compiler.export_python: could not bind the call arguments to "
                f"{getattr(self._fn, '__name__', 'fn')}'s signature: {e}"
            ) from e
        # bound.kwargs holds every argument bind() could not place positionally. That
        # is a keyword-only / **kwargs param (never positional), or a plain
        # positional-or-keyword param passed by keyword while an earlier one was left
        # to its default -- distinguish them so the error names the real cause.
        if bound.kwargs:
            params = sig.parameters
            kw_only = sorted(
                n
                for n in bound.kwargs
                if n in params and params[n].kind == inspect.Parameter.KEYWORD_ONLY
            )
            if kw_only:
                raise TypeError(
                    "torch.compiler.export_python does not support keyword-only "
                    f"parameters (got {kw_only}); the precompile calling convention "
                    "is positional."
                )
            # Names not declared as parameters were absorbed by a **kwargs param;
            # they are never positional, so name **kwargs as the cause rather than
            # misreporting them as a positional-or-keyword arg left to its default.
            var_kw = sorted(n for n in bound.kwargs if n not in params)
            if var_kw:
                raise TypeError(
                    "torch.compiler.export_python does not support **kwargs "
                    f"parameters (got {var_kw}); the precompile calling convention "
                    "is positional."
                )
            raise TypeError(
                "torch.compiler.export_python could not place keyword arguments "
                f"{sorted(bound.kwargs)} positionally because an earlier positional "
                "parameter was left to its default; pass those arguments positionally "
                "or provide example_inputs."
            )
        return bound.args

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs:
            args = self._bind_positional(args, kwargs)
        if self._loaded is None:
            with self._lock:
                if self._loaded is None:
                    self._loaded = self._materialize(args)
        return self._loaded(*args)


def export_python(
    *,
    path: str,
    backend: str = "inductor",
    tracer: str = "make_fx",
    decompositions: dict | None = None,
    example_inputs: Sequence[object] | None = None,
) -> Callable[[Callable[..., Any]], ExportedPythonArtifact]:
    """See :func:`torch.compiler.export_python`."""

    def decorator(fn: Callable[..., Any]) -> ExportedPythonArtifact:
        return ExportedPythonArtifact(
            fn,
            path=path,
            backend=backend,
            tracer=tracer,
            decompositions=decompositions,
            example_inputs=example_inputs,
        )

    return decorator
