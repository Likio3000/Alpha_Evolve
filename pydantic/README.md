This lightweight module provides just enough of the `pydantic` API for the
project's dashboard routes to run inside the execution environment used by the
tests. It is **not** a drop-in replacement for the real library, but implements
only the pieces that the application touches (basic model validation, `Field`,
and `ValidationError`).
