"""Loopback and cloud-tag guards — the two refusals that keep Hearth fully offline.

Both are checked before a request ever leaves the process, not caught afterward: a
provider that connected to a remote host and then refused the *response* has already sent
the prompt over the network, which is the one thing this project promises never to do.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

from hearth.llm.errors import ProviderConfigError

#: Hostnames that resolve to the local machine without needing DNS.
_LOOPBACK_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1"})

#: Suffixes on a model tag that mean "run this in the vendor's cloud, not locally."
_CLOUD_TAG_SUFFIXES = (":cloud", "-cloud")


def is_loopback_host(host: str) -> bool:
    """Whether a host string names the local machine.

    Bare IPv6 addresses are checked via :mod:`ipaddress` *before* URL parsing, because
    ``urlparse`` requires bracket syntax (``[::1]``) for a bare IPv6 literal and would
    otherwise misparse or reject one that a config file wrote unbracketed.
    """
    candidate = host.strip()
    if not candidate:
        return False

    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        pass

    # Not a bare IP literal — parse as a URL or a bare hostname[:port].
    parsed = urlparse(candidate if "//" in candidate else f"//{candidate}")
    hostname = parsed.hostname or candidate.split(":")[0]

    if hostname in _LOOPBACK_HOSTNAMES:
        return True

    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def ensure_loopback_host(host: str, *, allow_remote_host: bool = False) -> None:
    """Refuse a non-loopback host, unless explicitly overridden.

    ``allow_remote_host`` is the single, explicit opt-out from the loopback guarantee. It
    exists for the WSL2 mirrored-networking case where "loopback" legitimately means the
    Windows side, and is otherwise never set (docs/system-design.md §1, non-negotiable
    rule 1).
    """
    if allow_remote_host:
        return
    if not is_loopback_host(host):
        raise ProviderConfigError(
            f"refusing non-loopback Ollama host {host!r}. Hearth only talks to a local "
            "server; set ollama.allow_remote_host=true if you understand the risk."
        )


def is_cloud_model(model: str) -> bool:
    """Whether a model tag names one of Ollama's cloud-hosted models."""
    lowered = model.strip().lower()
    return any(lowered.endswith(suffix) for suffix in _CLOUD_TAG_SUFFIXES)


def ensure_not_cloud_model(model: str) -> None:
    """Refuse a cloud-tagged model. No override exists for this one.

    Unlike the loopback host, there is no legitimate reason for Hearth to ever address a
    cloud model — the whole premise is that nothing leaves the machine.
    """
    if is_cloud_model(model):
        raise ProviderConfigError(
            f"refusing cloud model {model!r}. Hearth only runs local models; "
            "remove the -cloud/:cloud suffix, or choose a local tag."
        )
