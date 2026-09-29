"""Strict source-byte bridge for unchanged LF/CRLF historical text files."""
import hashlib


def verify_source_bytes(working, git_blob, frozen_sha):
    normalized = lambda data: data.replace(b"\r\n", b"\n")
    digest = lambda data: hashlib.sha256(data).hexdigest()
    if normalized(working) != normalized(git_blob):
        raise ValueError("working source differs from committed source")
    if digest(working) == frozen_sha:
        return "exact"
    variants = (git_blob, normalized(git_blob), normalized(git_blob).replace(b"\n", b"\r\n"))
    if frozen_sha in {digest(data) for data in variants}:
        return "CRLF_only"
    raise ValueError("frozen source is not an exact Git byte variant")
