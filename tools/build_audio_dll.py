import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

NAME = 'libsndfile-1.2.2'
SHA256 = '3799ca9924d3125038880367bf1468e53a1b7e3686a934f098b7e1d286cdb80e'
URL = 'https://github.com/libsndfile/libsndfile/releases/download/1.2.2/libsndfile-1.2.2.tar.xz'
FLAGS = ['-DCMAKE_POLICY_VERSION_MINIMUM=3.5', '-DCMAKE_VS_GLOBALS=VcpkgEnabled=false', '-DCMAKE_TRY_COMPILE_PLATFORM_VARIABLES=CMAKE_VS_GLOBALS', '-DCMAKE_SHARED_LINKER_FLAGS_RELEASE=/INCREMENTAL:NO /DEBUG:NONE /Brepro', '-DBUILD_SHARED_LIBS=ON', '-DBUILD_PROGRAMS=OFF', '-DBUILD_EXAMPLES=OFF', '-DBUILD_TESTING=OFF', '-DENABLE_CPACK=OFF', '-DENABLE_EXTERNAL_LIBS=OFF', '-DENABLE_MPEG=OFF', '-DENABLE_EXPERIMENTAL=OFF'] + ['-DCMAKE_DISABLE_FIND_PACKAGE_' + name + '=TRUE' for name in ('FLAC', 'Vorbis', 'Opus', 'Ogg', 'mp3lame', 'mpg123', 'Speex', 'SQLite3')]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, default=Path('build/audio-native-min'))
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--cmake', default=shutil.which('cmake'))
    args = parser.parse_args()
    if not args.cmake:
        raise RuntimeError('CMAKE_REQUIRED')
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    archive = work / 'libsndfile.tar.xz'
    if args.archive and args.archive.resolve() != archive:
        shutil.copyfile(args.archive, archive)
    if not archive.is_file():
        with urllib.request.urlopen(URL, timeout=30) as response:
            data = response.read(2 * 1024 * 1024)
        archive.write_bytes(data)
    if digest(archive) != SHA256:
        raise RuntimeError('NATIVE_SOURCE_HASH_INVALID')
    source = work / NAME
    if source.exists():
        if source.resolve().parent != work:
            raise RuntimeError('NATIVE_SOURCE_PATH_INVALID')
        shutil.rmtree(source)
    with tarfile.open(archive, 'r:xz') as files:
        files.extractall(work, filter='data')
    build = work / 'native-build'
    subprocess.run([args.cmake, '-S', str(source), '-B', str(build), '-G', 'Visual Studio 17 2022', '-A', 'x64', *FLAGS], check=True)
    subprocess.run([args.cmake, '--build', str(build), '--config', 'Release', '--parallel', '8', '--', '/p:VcpkgEnabled=false'], check=True)
    dll = build / 'Release' / 'sndfile.dll'
    if not dll.is_file():
        raise RuntimeError('NATIVE_OUTPUT_MISSING')
    result = {'version': '1.2.2', 'archive': archive.name, 'source_sha256': SHA256, 'url': URL, 'dll_sha256': digest(dll), 'generator': 'Visual Studio 17 2022', 'arch': 'x64', 'flags': FLAGS, 'source_modified': False, 'external_codecs': False, 'cmake_version': subprocess.check_output([args.cmake, '--version'], text=True).splitlines()[0]}
    (work / 'native.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'dll': str(dll), 'sha256': result['dll_sha256']}))


if __name__ == '__main__':
    main()
