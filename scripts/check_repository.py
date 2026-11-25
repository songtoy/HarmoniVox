"""Check that existing Git source files match the inference release boundary."""

import subprocess

from package_release import FILES, ROOT


def main():
    allowed = set(FILES)
    allowed.update(p.name for pattern in ("LICENSE*", "NOTICE*")
                   for p in ROOT.glob(pattern) if p.is_file())
    listed = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    ).decode().split("\0")
    # During preparation, removed research files are still in the old index.
    # On CI a clean checkout naturally checks the committed source tree.
    existing = {name for name in listed if name and (ROOT / name).exists()}
    errors = [f"Unexpected source file: {name}" for name in sorted(existing - allowed)]
    errors += [f"Missing release file: {name}" for name in sorted(allowed - existing)]
    for name in sorted(existing):
        path = ROOT / name
        if path.is_symlink():
            errors.append(f"Source must not be a symlink: {name}")
        elif not path.is_file() or path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f"Unexpected directory or large source file: {name}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Repository boundary OK: {len(existing)} source files; weights stay outside Git.")


if __name__ == "__main__":
    main()
