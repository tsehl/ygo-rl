"""Provenance et fichiers de résultats immuables d'une campagne de validation."""
import hashlib
import json
from pathlib import Path
import platform
import os
import subprocess
from datetime import datetime, timezone
import uuid

ROOT = Path(__file__).resolve().parents[1]


def git_head(path):
    if not (Path(path) / ".git").exists():
        return None
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def fingerprint():
    # Inclut les modifications locales et le pilote, pas seulement le commit.
    files = {}
    for folder in ('engine', 'oracle', 'translate', 'train', 'gen'):
        for p in sorted([*(ROOT / folder).glob('*.py'), *(ROOT / folder).glob('*.cpp'), *(ROOT / folder).glob('*.lua')]):
            files[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted((ROOT / 'tests').rglob('*')):
        if p.is_file() and p.suffix in ('.py', '.lua', '.json'):
            files[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    for folder in ('oracle/ygopro-core', 'data/CardScripts', 'data/BabelCDB'):
        path = ROOT / folder
        files[folder] = git_head(path)
        # HEAD seul ne détecte pas les modifications locales des scripts/données.
        diff = subprocess.check_output(['git', '-C', str(path), 'diff', 'HEAD', '--binary'])
        files[folder + ':diff'] = hashlib.sha256(diff).hexdigest()
        untracked = subprocess.check_output(['git', '-C', str(path), 'ls-files', '--others', '--exclude-standard', '-z'])
        for name in untracked.decode().split('\0'):
            if name and (path / name).is_file():
                files[folder + '/' + name] = hashlib.sha256((path / name).read_bytes()).hexdigest()
    for name in ("oracle/ygopro-core/bin/release/libocgcore.so", "data/BabelCDB/cards.cdb", "oracle/ygopro-core/bin/release/libocgcore.dylib", ".tools/libeffect-inspector.so"):
        p = ROOT / name
        if p.exists():
            files[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return files


def package_versions():
    import importlib.metadata
    versions = {}
    for p in ("jax", "jaxlib", "numpy", "luaparser"):
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = None
    return versions


def provenance(args):
    files = fingerprint()
    versions = package_versions()
    identity = dict(files=files, versions=versions, python=platform.python_version(),
                    architecture=platform.machine(), runtime_config={k: os.environ.get(k)
                    for k in ("JAX_ENABLE_X64", "JAX_PLATFORMS", "XLA_FLAGS")})
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return dict(schema=2, signature=signature, commit=git_head(ROOT), identity=identity,
                args=vars(args), started_at=datetime.now(timezone.utc).isoformat())


def save_run(directory, manifest, games):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:12] + '.json')
    temporary = path.with_suffix('.json.tmp')
    try:
        with temporary.open('x') as f:
            json.dump(dict(manifest=manifest, games=games), f, indent=2)
        # Le lien est créé uniquement lorsque le JSON est complet, sans écraser un résultat existant.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


class RunJournal:
    """Conserve les parties terminées même si une campagne est interrompue."""
    def __init__(self, directory, manifest):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / (uuid.uuid4().hex + '.jsonl')
        with self.path.open('x') as f:
            f.write(json.dumps(dict(manifest=manifest)) + '\n')

    def append(self, game):
        with self.path.open('a') as f:
            f.write(json.dumps(dict(game=game)) + '\n')
