import glob
import platform
import shutil
from pathlib import Path
from subprocess import CalledProcessError

from anyio import run_process
from pydantic import BaseModel, ConfigDict


class LinuxMonitorError(Exception):
    """Exception raised for errors in Linux monitoring functions."""


def _check_platform() -> None:
    """
    Check if the current platform is Linux.

    Raises:
        LinuxMonitorError: If not running on Linux.
    """
    if platform.system().lower() != "linux":
        raise LinuxMonitorError("Linux monitor only supports Linux OS")


class TempMetrics(BaseModel):
    """Temperature-related metrics for Linux."""

    cpu_temp_avg: float = 0.0
    gpu_temp_avg: float = 0.0

    model_config = ConfigDict(extra="ignore")


class Metrics(BaseModel):
    """Complete set of metrics for Linux systems.

    Unknown fields are ignored for forward-compatibility.
    """

    all_power: float = 0.0
    ane_power: float = 0.0
    cpu_power: float = 0.0
    ecpu_usage: tuple[int, float] = (0, 0.0)
    gpu_power: float = 0.0
    gpu_ram_power: float = 0.0
    gpu_usage: tuple[int, float] = (0, 0.0)
    pcpu_usage: tuple[int, float] = (0, 0.0)
    ram_power: float = 0.0
    sys_power: float = 0.0
    temp: TempMetrics = TempMetrics()
    timestamp: str = ""

    model_config = ConfigDict(extra="ignore")


def _read_sys_file(path: str) -> str | None:
    """Read a system file and return its contents, or None if error."""
    try:
        return Path(path).read_text().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _get_cpu_temp() -> float:
    """
    Get CPU temperature from /sys/class/thermal or /sys/class/hwmon.

    Returns:
        CPU temperature in Celsius, or 0.0 if unavailable.
    """
    # Try thermal_zone files
    thermal_zones = glob.glob("/sys/class/thermal/thermal_zone*/temp")
    for zone in thermal_zones:
        temp_str = _read_sys_file(zone)
        if temp_str and temp_str.isdigit():
            # Temperature is in millidegrees Celsius
            temp = int(temp_str) / 1000.0
            # Sanity check: reasonable CPU temp range
            if 0 < temp < 120:
                return temp

    # Try hwmon sensors
    hwmon_temps = glob.glob("/sys/class/hwmon/hwmon*/temp*_input")
    for temp_file in hwmon_temps:
        # Check if this is a CPU-related sensor
        name_file = str(Path(temp_file).parent / "name")
        name = _read_sys_file(name_file)
        if name and ("coretemp" in name.lower() or "k10temp" in name.lower() or "cpu" in name.lower()):
            temp_str = _read_sys_file(temp_file)
            if temp_str and temp_str.isdigit():
                temp = int(temp_str) / 1000.0
                if 0 < temp < 120:
                    return temp

    return 0.0


async def _get_nvidia_smi_metrics() -> dict[str, float]:
    """
    Query NVIDIA GPU metrics using nvidia-smi.

    Returns:
        Dictionary with gpu_usage, gpu_temp, and gpu_power.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return {"gpu_usage": 0.0, "gpu_temp": 0.0, "gpu_power": 0.0}

    try:
        # Query GPU utilization, temperature, and power
        result = await run_process(
            [
                nvidia_smi,
                "--query-gpu=utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ]
        )

        output = result.stdout.decode().strip()
        if output:
            # Parse first GPU (support multi-GPU later)
            values = output.split("\n")[0].split(",")
            if len(values) >= 3:
                return {
                    "gpu_usage": float(values[0].strip()) / 100.0,  # Convert to 0-1
                    "gpu_temp": float(values[1].strip()),
                    "gpu_power": float(values[2].strip()),
                }
    except (CalledProcessError, ValueError, IndexError):
        pass

    return {"gpu_usage": 0.0, "gpu_temp": 0.0, "gpu_power": 0.0}


async def _get_amd_rocm_metrics() -> dict[str, float]:
    """
    Query AMD GPU metrics using rocm-smi (if available).

    Returns:
        Dictionary with gpu_usage, gpu_temp, and gpu_power.
    """
    rocm_smi = shutil.which("rocm-smi")
    if rocm_smi is None:
        return {"gpu_usage": 0.0, "gpu_temp": 0.0, "gpu_power": 0.0}

    try:
        # Query GPU usage
        result = await run_process([rocm_smi, "--showuse", "--csv"])
        usage_output = result.stdout.decode().strip()

        # Query GPU temperature
        temp_result = await run_process([rocm_smi, "--showtemp", "--csv"])
        temp_output = temp_result.stdout.decode().strip()

        # Query GPU power
        power_result = await run_process([rocm_smi, "--showpower", "--csv"])
        power_output = power_result.stdout.decode().strip()

        gpu_usage = 0.0
        gpu_temp = 0.0
        gpu_power = 0.0

        # Parse usage (simplified)
        if usage_output:
            lines = usage_output.split("\n")
            if len(lines) > 1:
                try:
                    usage_line = lines[1].split(",")
                    if len(usage_line) > 1:
                        gpu_usage = float(usage_line[1].strip().rstrip("%")) / 100.0
                except (ValueError, IndexError):
                    pass

        # Parse temperature
        if temp_output:
            lines = temp_output.split("\n")
            if len(lines) > 1:
                try:
                    temp_line = lines[1].split(",")
                    if len(temp_line) > 1:
                        gpu_temp = float(temp_line[1].strip())
                except (ValueError, IndexError):
                    pass

        # Parse power
        if power_output:
            lines = power_output.split("\n")
            if len(lines) > 1:
                try:
                    power_line = lines[1].split(",")
                    if len(power_line) > 1:
                        # Power might be in watts or milliwatts
                        power_str = power_line[1].strip().rstrip("W")
                        gpu_power = float(power_str)
                except (ValueError, IndexError):
                    pass

        return {"gpu_usage": gpu_usage, "gpu_temp": gpu_temp, "gpu_power": gpu_power}
    except (CalledProcessError, ValueError, IndexError):
        pass

    return {"gpu_usage": 0.0, "gpu_temp": 0.0, "gpu_power": 0.0}


async def get_metrics_async() -> Metrics:
    """
    Asynchronously gather Linux system metrics.

    Returns:
        A Metrics object containing system metrics.

    Raises:
        LinuxMonitorError: If there's an error gathering metrics.
    """
    _check_platform()

    from datetime import datetime

    import psutil

    # Try NVIDIA first, then AMD
    gpu_metrics = await _get_nvidia_smi_metrics()
    if gpu_metrics["gpu_usage"] == 0.0 and gpu_metrics["gpu_temp"] == 0.0:
        gpu_metrics = await _get_amd_rocm_metrics()

    # Get CPU usage
    cpu_percent = psutil.cpu_percent(interval=0.1) / 100.0

    # Get CPU temperature
    cpu_temp = _get_cpu_temp()

    return Metrics(
        gpu_usage=(0, gpu_metrics["gpu_usage"]),
        gpu_power=gpu_metrics["gpu_power"],
        pcpu_usage=(0, cpu_percent),
        ecpu_usage=(0, cpu_percent),
        temp=TempMetrics(
            gpu_temp_avg=gpu_metrics["gpu_temp"],
            cpu_temp_avg=cpu_temp,
        ),
        timestamp=datetime.now().isoformat(),
    )
