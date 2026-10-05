import hashlib
import importlib.metadata as metadata
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "progress-v1"
WORK = ROOT / "build" / "progress-v1"
BIN = ROOT / "build" / "progress-bin"
NATIVE = ROOT / "build" / "audio-native-min"
RUNTIME_IDS = ("numpy", "soundfile", "cffi", "pycparser")
LICENSE_IDS = {"numpy": "BSD-3-Clause", "soundfile": "BSD-3-Clause", "cffi": "MIT", "pycparser": "BSD-3-Clause"}
LIMITS = ["FLAC_UNAVAILABLE_PROGRESS", "GPU_UNAVAILABLE_PROGRESS", "SCIENTIFIC_WORKFLOWS_UNAVAILABLE_PROGRESS", "UNIFORM_63_DERIVATION_UNQUALIFIED", "AUTOMATED_PRIMARY_INTAKE_UNIMPLEMENTED", "PRIMARY_CORPUS_UNCERTIFIED"]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_zip(path, files, root):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source in sorted(files):
            info = zipfile.ZipInfo(source.relative_to(root).as_posix(), date_time=(2026, 10, 4, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes())


def main():
    if sys.version.split()[0] != (ROOT / ".python-version").read_text().strip():
        raise RuntimeError("BUILD_PYTHON_VERSION_MISMATCH")
    subprocess.run([sys.executable, str(ROOT / "build.py")], cwd=ROOT, check=True)
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    native_command = [sys.executable, str(ROOT / "tools" / "build_audio_dll.py"), "--cmake", shutil.which("cmake") or "C:/Program Files/CMake/bin/cmake.exe"]
    cached_archive = ROOT / "build" / "audio-native" / "libsndfile.tar.xz"
    if cached_archive.is_file():
        native_command.extend(["--archive", str(cached_archive)])
    subprocess.run(native_command, cwd=ROOT, check=True)
    apache = NATIVE / "apache2.license"
    if not apache.is_file():
        with urllib.request.urlopen("https://www.apache.org/licenses/LICENSE-2.0.txt", timeout=30) as response:
            apache.write_bytes(response.read(100000))
    if digest(apache) != "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30":
        raise RuntimeError("APACHE_LICENSE_HASH_INVALID")
    command = [sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm", "--noupx", "--onedir", "--console", "--name", "Chemosound", "--distpath", str(BIN), "--workpath", str(WORK), "--specpath", str(WORK), "--paths", str(ROOT / "src")]
    for name in ("cupy", "pytest", "matplotlib", "tkinter", "ssl", "_ssl", "_hashlib"):
        command.extend(["--exclude-module", name])
    command.append(str(ROOT / "public_start.py"))
    build_env = {**os.environ, "SOURCE_DATE_EPOCH": "1791072000", "PYTHONHASHSEED": "0"}
    subprocess.run(command, cwd=ROOT, env=build_env, check=True)
    bundle = BIN / "Chemosound"
    native = json.loads((NATIVE / "native.json").read_text())
    dll = NATIVE / "native-build" / "Release" / "sndfile.dll"
    if digest(dll) != native["dll_sha256"]:
        raise RuntimeError("NATIVE_OUTPUT_HASH_INVALID")
    dll_target = bundle / "_internal" / "_soundfile_data" / "libsndfile_x64.dll"
    shutil.copyfile(dll, dll_target)
    for path in (bundle / "_internal").glob("*.dist-info/RECORD"):
        path.unlink()
    license_root = bundle / "licenses"
    license_root.mkdir(exist_ok=True)
    dependencies = []
    for name in RUNTIME_IDS:
        dist = metadata.distribution(name)
        notices = []
        for index, item in enumerate(dist.files or []):
            if "license" not in item.name.lower() and "copying" not in item.name.lower():
                continue
            source = Path(dist.locate_file(item))
            if not source.is_file():
                continue
            target = license_root / (name + "-" + str(index) + ".license")
            shutil.copyfile(source, target)
            notices.append({"path": target.relative_to(bundle).as_posix(), "sha256": digest(target)})
        dependencies.append({"id": name, "version": dist.version, "license": LICENSE_IDS[name], "notices": notices})
    source_root = NATIVE / "libsndfile-1.2.2"
    notices = []
    for name, source in (("libsndfile.license", source_root / "COPYING"), ("libsndfile.authors", source_root / "AUTHORS"), ("gsm610.notice", source_root / "src" / "GSM610" / "COPYRIGHT"), ("apache2.license", NATIVE / "apache2.license")):
        target = license_root / name
        shutil.copyfile(source, target)
        notices.append({"path": target.relative_to(bundle).as_posix(), "sha256": digest(target)})
    for name, source in (("g72x.notice", source_root / "src" / "G72x" / "g72x.c"), ("alac.notice", source_root / "src" / "ALAC" / "alac_decoder.c"), ("alac_types.notice", source_root / "src" / "ALAC" / "ALACAudioTypes.h")):
        data = source.read_bytes()
        end = data.index(b"*/") + 2
        target = license_root / name
        target.write_bytes(data[:end] + b"\n")
        notices.append({"path": target.relative_to(bundle).as_posix(), "sha256": digest(target)})
    corr = bundle / "corresponding_source"
    corr.mkdir(exist_ok=True)
    for source in (NATIVE / "libsndfile.tar.xz", NATIVE / "native.json", ROOT / "tools" / "build_audio_dll.py"):
        shutil.copyfile(source, corr / source.name)
    dependencies.append({"id": "libsndfile", "version": "1.2.2", "license": "LGPL-2.1-or-later", "notices": notices, "dll": dll_target.relative_to(bundle).as_posix(), "dll_sha256": digest(dll_target), "source": "corresponding_source/libsndfile.tar.xz", "source_sha256": native["source_sha256"], "source_modified": False, "external_codecs": False, "linkage": "dynamic", "replacement_allowed": True, "reverse_engineering_debug_modifications_allowed": True, "build_command": ["python", "corresponding_source/build_audio_dll.py", "--archive", "corresponding_source/libsndfile.tar.xz"]})
    source = Path(sys.base_prefix) / "LICENSE.txt"
    if not source.is_file():
        raise RuntimeError("PYTHON_LICENSE_MISSING")
    target = license_root / "python.license"
    shutil.copyfile(source, target)
    dependencies.append({"id": "cpython", "version": sys.version.split()[0], "license": "PSF-2.0", "notices": [{"path": target.relative_to(bundle).as_posix(), "sha256": digest(target)}]})
    pyi = metadata.distribution("pyinstaller")
    target = license_root / "pyinstaller.license"
    license_file = next(Path(pyi.locate_file(item)) for item in pyi.files if item.name == "COPYING.txt")
    shutil.copyfile(license_file, target)
    dependencies.append({"id": "pyinstaller_boot", "version": pyi.version, "license": "GPL-2.0-or-later WITH Bootloader-exception", "notices": [{"path": target.relative_to(bundle).as_posix(), "sha256": digest(target)}]})
    save_json(bundle / "dependencies.json", dependencies)
    save_json(bundle / "requirements.json", {"os": "windows", "os_min": "10", "arch": "x86_64", "release_stage": "progress", "backends": ["cpu"], "formats": ["WAV", "AIFF"], "subtypes": ["PCM_S8", "PCM_U8", "PCM_16", "PCM_24", "PCM_32"], "gpu": False, "hdr": False, "admin": False, "reboot": False, "network": {"local_workflows": False, "acquisition": False}, "runtime_bundled": True, "datasets_bundled": False, "primary_inference_eligible": False, "commands": ["runtime", "init", "scan", "run", "status", "retry-failed", "freeze"], "limitation_codes": LIMITS, "native_build_requirements": {"cmake": "3.21+", "compiler": "Visual Studio 2022 C++ Build Tools", "python": "3.14+"}})
    source_files = list((ROOT / "src" / "chemosound").glob("*.py"))
    source_files.extend((ROOT / "tools").glob("*.py"))
    source_files.extend((ROOT / "research").rglob("*.py"))
    source_files.extend((ROOT / "research" / "acquisition").glob("*.json"))
    source_files.extend((ROOT / "research" / "acquisition").glob("*.sql"))
    source_files.extend(ROOT / name for name in ("pyproject.toml", "uv.lock", "build.py", "build_hook.py", "public_start.py", "tests/are_comments.py", ".python-version", ".gitignore"))
    save_json(OUT / "source-files.json", [{"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path), "bytes": path.stat().st_size} for path in sorted(source_files)])
    write_zip(OUT / "source.zip", source_files, ROOT)
    binary_files = [path for path in bundle.rglob("*") if path.is_file()]
    save_json(OUT / "binary-files.json", [{"path": path.relative_to(bundle).as_posix(), "sha256": digest(path), "bytes": path.stat().st_size} for path in sorted(binary_files)])
    write_zip(OUT / "portable.zip", binary_files, BIN)
    save_json(OUT / "hashes.json", {path.name: digest(path) for path in (OUT / "source.zip", OUT / "portable.zip", bundle / "Chemosound.exe")})
    print(json.dumps({"package": str(OUT / "portable.zip"), "source_snapshot": str(OUT / "source.zip"), "executable": str(bundle / "Chemosound.exe")}, sort_keys=True))


if __name__ == "__main__":
    main()
