import sys
from pathlib import Path
from pydub import AudioSegment
from mutagen import File as MutagenFile

SUPPORTED = {".mp3", ".wav"}

def validate(path_str: str):
    p = Path(path_str)
    if not p.exists() or not p.is_file():
        return False, "Файл не існує або це не звичайний файл"
    ext = p.suffix.lower()
    if ext not in SUPPORTED:
        return False, f"Непідтримуваний формат: {ext}. Підтримуються лише .mp3 та .wav"
    return True, ext

def duration_seconds(path: str, ext: str):
    try:
        fmt = ext.lstrip(".")
        audio = AudioSegment.from_file(path, format=fmt)
        return len(audio) / 1000.0
    except Exception:
        return None

def metadata_dict(path: str):
    data = {}
    try:
        m = MutagenFile(path, easy=False)
        if not m:
            return data
        if getattr(m, "tags", None):
            for k, v in m.tags.items():
                if hasattr(v, "text"):
                    data[str(k)] = ", ".join(map(str, v.text))
                else:
                    data[str(k)] = ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)
        if getattr(m, "info", None):
            info = m.info
            if getattr(info, "bitrate", None):
                data["bitrate"] = f"{info.bitrate} bps"
            if getattr(info, "sample_rate", None):
                data["sample_rate"] = f"{info.sample_rate} Hz"
            if getattr(info, "channels", None):
                data["channels"] = str(info.channels)
    except Exception:
        pass
    return data

def analyze(path: str):
    ok, ext_or_msg = validate(path)
    if not ok:
        print(f"Помилка: {ext_or_msg}")
        return 1

    ext = ext_or_msg
    print(f"Файл: {path}")
    print(f"Формат: {ext.upper()}")

    d = duration_seconds(path, ext)
    if d is None:
        print("Тривалість: неможливо визначити")
    else:
        print(f"Тривалість: {d:.2f} секунд")

    md = metadata_dict(path)
    if md:
        print("Метадані:")
        for k in sorted(md):
            print(f"  {k}: {md[k]}")
    else:
        print("Метадані відсутні")

    return 0

def main():
    if len(sys.argv) != 2:
        print("Використання: python media_info.py <шлях_до_файла.mp3|.wav>")
        sys.exit(2)
    sys.exit(analyze(sys.argv[1]))

if __name__ == "__main__":
    main()
