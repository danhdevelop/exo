import socket
import sys
from subprocess import CalledProcessError

import psutil
from anyio import run_process

from exo.shared.types.profiling import InterfaceType, NetworkInterfaceInfo


async def get_friendly_name() -> str:
    """
    Asynchronously gets the 'Computer Name' (friendly name) of a Mac.
    e.g., "John's MacBook Pro"
    Returns the name as a string, or None if an error occurs or not on macOS.
    """
    hostname = socket.gethostname()

    if sys.platform != "darwin":
        return hostname

    try:
        process = await run_process(["scutil", "--get", "ComputerName"])
    except CalledProcessError:
        return hostname

    return process.stdout.decode("utf-8", errors="replace").strip() or hostname


async def _get_interface_types_from_networksetup() -> dict[str, InterfaceType]:
    """Parse networksetup -listallhardwareports to get interface types."""
    if sys.platform != "darwin":
        return {}

    try:
        result = await run_process(["networksetup", "-listallhardwareports"])
    except CalledProcessError:
        return {}

    types: dict[str, InterfaceType] = {}
    current_type: InterfaceType = "unknown"

    for line in result.stdout.decode().splitlines():
        if line.startswith("Hardware Port:"):
            port_name = line.split(":", 1)[1].strip()
            if "Wi-Fi" in port_name:
                current_type = "wifi"
            elif "Ethernet" in port_name or "LAN" in port_name:
                current_type = "ethernet"
            elif port_name.startswith("Thunderbolt"):
                current_type = "thunderbolt"
            else:
                current_type = "unknown"
        elif line.startswith("Device:"):
            device = line.split(":", 1)[1].strip()
            # enX is ethernet adapters or thunderbolt - these must be deprioritised
            if device.startswith("en") and device not in ["en0", "en1"]:
                current_type = "maybe_ethernet"
            types[device] = current_type

    return types


async def get_network_interfaces() -> list[NetworkInterfaceInfo]:
    """
    Retrieves detailed network interface information on macOS.
    Parses output from 'networksetup -listallhardwareports' and 'ifconfig'
    to determine interface names, IP addresses, and types (ethernet, wifi, vpn, other).
    Returns a list of NetworkInterfaceInfo objects.
    """
    interfaces_info: list[NetworkInterfaceInfo] = []
    interface_types = await _get_interface_types_from_networksetup()

    for iface, services in psutil.net_if_addrs().items():
        for service in services:
            match service.family:
                case socket.AF_INET | socket.AF_INET6:
                    interfaces_info.append(
                        NetworkInterfaceInfo(
                            name=iface,
                            ip_address=service.address,
                            interface_type=interface_types.get(iface, "unknown"),
                        )
                    )
                case _:
                    pass

    return interfaces_info


async def get_model_and_chip() -> tuple[str, str]:
    """Get system information based on platform."""
    model = "Unknown Model"
    chip = "Unknown Chip"

    if sys.platform == "darwin":
        # macOS
        try:
            process = await run_process(
                [
                    "system_profiler",
                    "SPHardwareDataType",
                ]
            )
            output = process.stdout.decode().strip()

            model_line = next(
                (line for line in output.split("\n") if "Model Name" in line), None
            )
            model = model_line.split(": ")[1] if model_line else "Unknown Model"

            chip_line = next((line for line in output.split("\n") if "Chip" in line), None)
            chip = chip_line.split(": ")[1] if chip_line else "Unknown Chip"
        except CalledProcessError:
            pass

    elif sys.platform.startswith("win"):
        # Windows
        try:
            # Get computer model
            process = await run_process(
                ["wmic", "computersystem", "get", "model"],
                shell=True
            )
            output = process.stdout.decode().strip()
            lines = [line.strip() for line in output.split("\n") if line.strip()]
            if len(lines) > 1:
                model = lines[1]

            # Get CPU info
            cpu_process = await run_process(
                ["wmic", "cpu", "get", "name"],
                shell=True
            )
            cpu_output = cpu_process.stdout.decode().strip()
            cpu_lines = [line.strip() for line in cpu_output.split("\n") if line.strip()]
            if len(cpu_lines) > 1:
                chip = cpu_lines[1]
        except (CalledProcessError, IndexError):
            pass

    elif sys.platform.startswith("linux"):
        # Linux
        try:
            # Get model from DMI
            try:
                process = await run_process(
                    ["cat", "/sys/devices/virtual/dmi/id/product_name"]
                )
                model = process.stdout.decode().strip()
            except CalledProcessError:
                pass

            # Get CPU info from /proc/cpuinfo
            try:
                process = await run_process(["cat", "/proc/cpuinfo"])
                output = process.stdout.decode().strip()
                for line in output.split("\n"):
                    if line.startswith("model name"):
                        chip = line.split(":", 1)[1].strip()
                        break
            except CalledProcessError:
                pass
        except (CalledProcessError, IndexError):
            pass

    return (model, chip)
