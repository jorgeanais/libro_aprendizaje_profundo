#!/usr/bin/env python3
"""
Inserta un badge "Open in Colab" como primera celda de cada notebook .ipynb.

Uso:
    python add_colab_badge.py                     # procesa todos los .ipynb del repo git actual
    python add_colab_badge.py cap1/ cap2/          # solo esas carpetas/archivos
    python add_colab_badge.py --user jorge --repo libro_aprendizaje_profundo --branch main

Por defecto detecta usuario/repo/rama automáticamente desde `git remote` y `git branch`,
así que normalmente basta con correrlo sin argumentos desde la raíz del repo.

Es idempotente: si un notebook ya tiene el badge (detecta el link a colab.research.google.com
en la primera celda), lo salta sin duplicar.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def detect_repo_root():
    root = run(["git", "rev-parse", "--show-toplevel"])
    return Path(root) if root else None


def detect_user_repo():
    url = run(["git", "remote", "get-url", "origin"])
    if not url:
        return None, None
    # soporta https://github.com/user/repo.git y git@github.com:user/repo.git
    m = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(\.git)?$", url)
    if m:
        return m.group(1), m.group(2)
    return None, None


def detect_branch():
    branch = run(["git", "branch", "--show-current"])
    return branch or "main"


# Carpetas que nunca deben tocarse aunque no estén en .gitignore
EXCLUDED_DIR_NAMES = {"_build", "node_modules", ".git", "venv", ".venv", "env", "__pycache__"}


def is_excluded(path):
    return any(part in EXCLUDED_DIR_NAMES for part in path.parts)


def find_notebooks_via_git(repo_root):
    """Usa `git ls-files` para listar solo notebooks trackeados o trackeables
    (respeta .gitignore automáticamente, así nunca toca _build/, venv/, etc.)."""
    out = run([
        "git", "-C", str(repo_root),
        "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.ipynb",
    ])
    if out is None:
        return None
    files = [repo_root / line for line in out.splitlines() if line.strip()]
    return sorted(f for f in files if not is_excluded(f.relative_to(repo_root)))


def find_notebooks(paths, repo_root):
    if not paths:
        via_git = find_notebooks_via_git(repo_root)
        if via_git is not None:
            return via_git
        # fallback si no hay git disponible: recorre el disco pero excluye carpetas conocidas
        search_root = repo_root or Path(".")
        return sorted(
            p for p in search_root.rglob("*.ipynb")
            if not is_excluded(p.relative_to(search_root))
        )
    result = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            result.extend(
                sorted(f for f in p.rglob("*.ipynb") if not is_excluded(f.relative_to(p)))
            )
        elif p.suffix == ".ipynb":
            result.append(p)
    return [f for f in result if not is_excluded(f)]


def cell_source(cell):
    return "".join(cell.get("source", []))


def has_frontmatter(nb):
    """El frontmatter YAML de MyST (título, autor, licencia, etc.) debe ser
    literalmente la primera celda, o MyST no lo reconoce. Lo detectamos para
    insertar el badge DESPUÉS de él en vez de antes."""
    cells = nb.get("cells", [])
    if not cells:
        return False
    first = cells[0]
    if first.get("cell_type") not in ("markdown", "raw"):
        return False
    return cell_source(first).lstrip().startswith("---")


def insertion_index(nb):
    return 1 if has_frontmatter(nb) else 0


def already_has_badge(nb):
    cells = nb.get("cells", [])
    idx = insertion_index(nb)
    if idx >= len(cells):
        return False
    candidate = cells[idx]
    if candidate.get("cell_type") != "markdown":
        return False
    return "colab.research.google.com" in cell_source(candidate)


def make_badge_cell(colab_url):
    line = (
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"({colab_url})"
    )
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line],
    }


def process_notebook(nb_path, repo_root, user, repo, branch, dry_run):
    rel_path = nb_path.resolve().relative_to(repo_root.resolve()).as_posix()
    colab_url = f"https://colab.research.google.com/github/{user}/{repo}/blob/{branch}/{rel_path}"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    if already_has_badge(nb):
        print(f"  [=] ya tiene badge, se omite: {rel_path}")
        return False

    badge_cell = make_badge_cell(colab_url)
    nb.setdefault("cells", []).insert(insertion_index(nb), badge_cell)

    if dry_run:
        print(f"  [dry-run] agregaría badge: {rel_path} -> {colab_url}")
        return True

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")

    print(f"  [+] badge agregado: {rel_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="Notebooks o carpetas a procesar (default: todo el repo)")
    parser.add_argument("--user", help="Usuario u organización de GitHub (auto-detectado si se omite)")
    parser.add_argument("--repo", help="Nombre del repositorio (auto-detectado si se omite)")
    parser.add_argument("--branch", help="Rama (auto-detectada si se omite, default 'main')")
    parser.add_argument("--dry-run", action="store_true", help="Muestra qué haría sin modificar archivos")
    args = parser.parse_args()

    repo_root = detect_repo_root()
    if repo_root is None:
        print("Error: no se detectó un repositorio git. Corre este script dentro del repo.", file=sys.stderr)
        sys.exit(1)

    user, repo = args.user, args.repo
    if not user or not repo:
        auto_user, auto_repo = detect_user_repo()
        user = user or auto_user
        repo = repo or auto_repo

    if not user or not repo:
        print(
            "Error: no se pudo detectar usuario/repo desde 'git remote'. "
            "Especifícalos con --user y --repo.",
            file=sys.stderr,
        )
        sys.exit(1)

    branch = args.branch or detect_branch()

    notebooks = find_notebooks(args.paths, repo_root)
    if not notebooks:
        print("No se encontraron notebooks (.ipynb).")
        return

    print(f"Repo: {user}/{repo}  Rama: {branch}  Notebooks encontrados: {len(notebooks)}\n")

    changed = 0
    for nb_path in notebooks:
        if process_notebook(nb_path, repo_root, user, repo, branch, args.dry_run):
            changed += 1

    print(f"\nListo. {changed} notebook(s) modificado(s) de {len(notebooks)}.")


if __name__ == "__main__":
    main()
