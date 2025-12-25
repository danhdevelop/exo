"""Tests for Windows monitor error handling.

These tests verify that Windows monitor errors are handled gracefully without
crashing the application or spamming logs.
"""

import platform
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.worker.utils.windows_monitor import WindowsMonitorError, get_metrics_async


@pytest.mark.skipif(
    platform.system().lower() != "windows",
    reason="Windows monitor only supports Windows OS",
)
class TestWindowsMonitorErrorHandling:
    """Test Windows monitor error handling."""

    async def test_platform_check(self) -> None:
        """Should raise WindowsMonitorError if not on Windows."""
        with patch("exo.worker.utils.windows_monitor.platform.system", return_value="Linux"):
            with pytest.raises(WindowsMonitorError) as exc_info:
                await get_metrics_async()

            assert "Windows monitor only supports Windows OS" in str(exc_info.value)

    async def test_metrics_with_nvidia_gpu(self) -> None:
        """Should successfully get metrics when nvidia-smi is available."""
        mock_result = MagicMock()
        mock_result.stdout.decode.return_value = "50, 65, 120.5\n"

        with (
            patch("exo.worker.utils.windows_monitor.shutil.which", return_value="nvidia-smi.exe"),
            patch("exo.worker.utils.windows_monitor.run_process", new_callable=AsyncMock) as mock_run,
            patch("exo.worker.utils.windows_monitor.psutil.cpu_percent", return_value=45.0),
        ):
            mock_run.return_value = mock_result

            metrics = await get_metrics_async()

            assert metrics.gpu_usage[1] == 0.5  # 50% -> 0.5
            assert metrics.temp.gpu_temp_avg == 65.0
            assert metrics.gpu_power == 120.5

    async def test_metrics_fallback_no_gpu(self) -> None:
        """Should return default metrics when no GPU tools are available."""
        with (
            patch("exo.worker.utils.windows_monitor.shutil.which", return_value=None),
            patch("exo.worker.utils.windows_monitor.psutil.cpu_percent", return_value=30.0),
        ):
            metrics = await get_metrics_async()

            assert metrics.gpu_usage[1] == 0.0
            assert metrics.temp.gpu_temp_avg == 0.0
            assert metrics.gpu_power == 0.0
            assert metrics.pcpu_usage[1] == 0.3  # 30% -> 0.3
