import platform
import shutil
from subprocess import CalledProcessError

from anyio import run_process
from pydantic import BaseModel, ConfigDict


class WindowsMonitorError(Exception):
    """Exception raised for errors in Windows monitoring functions."""


def _check_platform() -> None:
    """
    Check if the current platform is Windows.

    Raises:
        WindowsMonitorError: If not running on Windows.
    """
    if platform.system().lower() != "windows":
        raise WindowsMonitorError("Windows monitor only supports Windows OS")


class TempMetrics(BaseModel):
    """Temperature-related metrics for Windows."""

    cpu_temp_avg: float = 0.0
    gpu_temp_avg: float = 0.0

    model_config = ConfigDict(extra="ignore")


class Metrics(BaseModel):
    """Complete set of metrics for Windows systems.

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

        gpu_usage = 0.0
        gpu_temp = 0.0

        # Parse usage (simplified)
        if usage_output:
            lines = usage_output.split("\n")
            if len(lines) > 1:
                # Try to extract usage percentage from CSV
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

        return {"gpu_usage": gpu_usage, "gpu_temp": gpu_temp, "gpu_power": 0.0}
    except (CalledProcessError, ValueError, IndexError):
        pass

    return {"gpu_usage": 0.0, "gpu_temp": 0.0, "gpu_power": 0.0}


async def get_metrics_async() -> Metrics:
    """
    Asynchronously gather Windows system metrics.

    Returns:
        A Metrics object containing system metrics.

    Raises:
        WindowsMonitorError: If there's an error gathering metrics.
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

    return Metrics(
        gpu_usage=(0, gpu_metrics["gpu_usage"]),
        gpu_power=gpu_metrics["gpu_power"],
        pcpu_usage=(0, cpu_percent),
        ecpu_usage=(0, cpu_percent),
        temp=TempMetrics(
            gpu_temp_avg=gpu_metrics["gpu_temp"],
            cpu_temp_avg=0.0,  # CPU temp requires WMI or OpenHardwareMonitor
        ),
        timestamp=datetime.now().isoformat(),
    )
