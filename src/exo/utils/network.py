"""Network interface auto-detection utilities for P2P traffic."""

from loguru import logger

from exo.utils.info_gatherer.system_info import get_network_interfaces


async def get_thunderbolt_interface_ip() -> str | None:
    """Auto-detect Thunderbolt interface IP address.

    Returns the IP address of the first active Thunderbolt interface,
    or None if no Thunderbolt interface is found.

    Only returns IPv4 addresses (excludes IPv6 and loopback).
    """
    interfaces = await get_network_interfaces()

    # Filter for Thunderbolt interfaces with valid IPv4 addresses
    for iface in interfaces:
        if iface.interface_type == "thunderbolt":
            # Validate it's IPv4 (not IPv6, not loopback)
            ip = iface.ip_address
            if ":" not in ip and not ip.startswith("127."):
                logger.debug(
                    f"Found Thunderbolt interface {iface.name} with IP {ip}"
                )
                return ip

    return None
