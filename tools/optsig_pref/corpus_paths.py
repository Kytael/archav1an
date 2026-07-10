"""Size-match the optsig-pref corpus (21 clips) to their gpu1 paths.

The same MVI_ numbers repeat across multiple folders under /mnt/media and
/mnt/media2, so files are identified by exact byte size (all 21 corpus
sizes are distinct).
"""
import subprocess

SEARCH_ROOTS = ["/mnt/media", "/mnt/media2"]
FIND_TIMEOUT_S = 120

# corpus size -> stem (authoritative, from encoder-host ~/batch19/src)
CORPUS = {
    1014517852: "1352", 1054575280: "0909", 1080022184: "7052", 1081286664: "5042",
    1204135748: "4165", 1222401472: "1455", 1556341604: "5686", 156790908: "8742",
    1597980456: "8267", 1696720928: "1389", 1823591672: "4849", 1880882776: "8656",
    382188188: "6174", 465247640: "5280", 627502568: "0265", 628797364: "4378",
    730561880: "3265", 78849188: "4848", 790356476: "5863", 918598796: "5281", 958318200: "0487",
}


def resolve_corpus():
    """Return [(stem, path), ...] for the 21 corpus clips, sorted by stem.

    Raises RuntimeError if any stem has no size-matched gpu1 file.
    """
    stems = sorted(CORPUS.values())
    alt = "\\|".join(stems)
    pattern = f".*MVI_\\({alt}\\).MOV"
    out = subprocess.check_output(
        ["find", *SEARCH_ROOTS, "-type", "f", "-iregex", pattern, "-printf", "%s\t%p\n"],
        text=True, timeout=FIND_TIMEOUT_S,
    )

    by_size = {}
    for line in out.splitlines():
        size_s, path = line.split("\t", 1)
        by_size.setdefault(int(size_s), path)

    result = []
    missing = []
    for size, stem in CORPUS.items():
        path = by_size.get(size)
        if path is None:
            missing.append(stem)
        else:
            result.append((stem, path))

    if missing:
        raise RuntimeError(f"no size-matched gpu1 file for stems: {sorted(missing)}")

    return sorted(result)
