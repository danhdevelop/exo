"""Tests for Linux monitor error handling.

These tests verify that Linux monitor errors are handled gracefully without
crashing the application or spamming logs.
"""

import platform
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.worker.utils.linux_monitor import LinuxMonitorError, get_metrics_async


@pytest.mark.skipif(
    platform.system().lower() != "linux",
    reason="Linux monitor only supports Linux OS",
)
class TestLinuxMonitorErrorHandling:
    """Test Linux monitor error handling."""

    async def test_platform_check(self) -> None:
        """Should raise LinuxMonitorError if not on Linux."""
        with patch("exo.worker.utils.linux_monitor.platform.system", return_value="Darwin"):
            with pytest.raises(LinuxMonitorError) as exc_info:
                await get_metrics_async()

            assert "Linux monitor only supports Linux OS" in str(exc_info.value)

    async def test_metrics_with_nvidia_gpu(self) -> None:
        """Should successfully get metrics when nvidia-smi is available."""
        mock_result = MagicMock()
        mock_result.stdout.decode.return_value = "75, 68, 150.2\n"

        with (
            patch("exo.worker.utils.linux_monitor.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("exo.worker.utils.linux_monitor.run_process", new_callable=AsyncMock) as mock_run,
            patch("exo.worker.utils.linux_monitor.psutil.cpu_percent", return_value=55.0),
            patch("exo.worker.utils.linux_monitor._get_cpu_temp", return_value=45.5),
        ):
            mock_run.return_value = mock_result

            metrics = await get_metrics_async()

            assert metrics.gpu_usage[1] == 0.75  # 75% -> 0.75
            assert metrics.temp.gpu_temp_avg == 68.0
            assert metrics.gpu_power == 150.2
            assert metrics.temp.cpu_temp_avg == 45.5

    async def test_metrics_fallback_no_gpu(self) -> None:
        """Should return default metrics when no GPU tools are available."""
        with (
            patch("exo.worker.utils.linux_monitor.shutil.which", return_value=None),
            patch("exo.worker.utils.linux_monitor.psutil.cpu_percent", return_value=40.0),
            patch("exo.worker.utils.linux_monitor._get_cpu_temp", return_value=50.0),
        ):
            metrics = await get_metrics_async()

            assert metrics.gpu_usage[1] == 0.0
            assert metrics.temp.gpu_temp_avg == 0.0
            assert metrics.gpu_power == 0.0
            assert metrics.pcpu_usage[1] == 0.4  # 40% -> 0.4
            assert metrics.temp.cpu_temp_avg == 50.0
