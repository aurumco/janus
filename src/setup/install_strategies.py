"""Installation strategies for different environments using Strategy Pattern.

This module implements the Strategy Pattern to handle package installation
across different environments (Kaggle, local, etc.) with clean separation
of concerns and extensibility.
"""

import os
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional


class InstallationStrategy(ABC):
    """Abstract base class for package installation strategies.
    
    This defines the interface that all concrete installation strategies
    must implement, following the Strategy Pattern.
    """

    @abstractmethod
    def install_core_packages(self) -> None:
        """Install core scientific computing packages.
        
        Raises:
            subprocess.CalledProcessError: If installation fails.
        """
        pass

    @abstractmethod
    def install_ml_packages(self) -> None:
        """Install machine learning specific packages.
        
        Raises:
            subprocess.CalledProcessError: If installation fails.
        """
        pass

    @abstractmethod
    def install_from_requirements(self, requirements_path: Path) -> None:
        """Install packages from requirements.txt file.
        
        Args:
            requirements_path: Path to requirements.txt file.
            
        Raises:
            subprocess.CalledProcessError: If installation fails.
        """
        pass

    def _run_pip_command(
        self,
        args: List[str],
        check: bool = True,
        capture_output: bool = True,
        env: Optional[dict] = None,
    ) -> subprocess.CompletedProcess:
        """Execute a pip command with standard error handling.
        
        Args:
            args: List of pip command arguments.
            check: Whether to raise exception on non-zero exit.
            capture_output: Whether to capture stdout/stderr.
            
        Returns:
            CompletedProcess instance with command results.
            
        Raises:
            subprocess.CalledProcessError: If check=True and command fails.
        """
        cmd = [sys.executable, "-m", "pip"] + args
        return subprocess.run(cmd, check=check, capture_output=capture_output, env=env)

    def _is_package_installed(self, package_name: str, min_version: Optional[str] = None) -> bool:
        """Check if a package is installed with optional version check.
        
        Args:
            package_name: Name of the package to check.
            min_version: Optional minimum version required.
            
        Returns:
            True if package is installed (and meets version requirement).
        """
        try:
            result = self._run_pip_command(
                ["show", package_name],
                check=False,
                capture_output=True
            )
            if result.returncode != 0:
                return False
            
            if min_version:
                output = result.stdout.decode("utf-8")
                for line in output.split("\n"):
                    if line.startswith("Version:"):
                        installed_version = line.split(":")[1].strip()
                        from packaging import version
                        return version.parse(installed_version) >= version.parse(min_version)
            return True
        except Exception:
            return False


class KaggleInstallationStrategy(InstallationStrategy):
    """Installation strategy optimized for Kaggle environment.
    
    Kaggle has pre-installed packages that may conflict with our requirements.
    This strategy handles those conflicts by:
    1. Uninstalling conflicting packages
    2. Installing compatible versions with proper flags
    3. Installing mamba-ssm and other ML packages separately
    """

    def __init__(self, verbose: bool = True) -> None:
        """Initialize Kaggle installation strategy.
        
        Args:
            verbose: Whether to print progress messages.
        """
        self.verbose = verbose

    def _log(self, message: str) -> None:
        """Log a message if verbose mode is enabled.
        
        Args:
            message: Message to log.
        """
        if self.verbose:
            print(f"[setup] {message}")

    def install_core_packages(self) -> None:
        self._log("Skipping core package installation (handled via requirements.txt)")

    def install_ml_packages(self) -> None:
        self._log("Skipping ML package installation (handled via requirements.txt)")

    def install_from_requirements(self, requirements_path: Path) -> None:
        """Install remaining packages from requirements.txt.
        
        Args:
            requirements_path: Path to requirements.txt file.
        """
        if not requirements_path.exists():
            self._log(f"Requirements file not found: {requirements_path}")
            return

        self._log("Installing packages from requirements.txt...")

        gpu_env = os.environ.copy()
        gpu_env.setdefault("FORCE_CUDA", "1")
        gpu_env.setdefault("MAX_JOBS", "4")
        conda_prefix = gpu_env.get("CONDA_PREFIX")
        if conda_prefix:
            lib_path = os.path.join(conda_prefix, "lib")
            old_ld = gpu_env.get("LD_LIBRARY_PATH", "")
            gpu_env["LD_LIBRARY_PATH"] = f"{lib_path}:{old_ld}" if old_ld else lib_path

        try:
            self._run_pip_command(
                [
                    "install",
                    "--upgrade",
                    "--requirement",
                    str(requirements_path),
                    "--extra-index-url",
                    "https://download.pytorch.org/whl/cu121",
                ],
                capture_output=False,
                env=gpu_env,
            )
            self._log("Requirements installed successfully")
        except subprocess.CalledProcessError as e:
            self._log(f"Warning: Failed to install requirements: {e}")


class LocalInstallationStrategy(InstallationStrategy):
    """Installation strategy for local development environment.
    
    This strategy assumes a clean environment and installs all packages
    directly from requirements.txt without special handling.
    """

    def __init__(self, verbose: bool = True) -> None:
        """Initialize local installation strategy.
        
        Args:
            verbose: Whether to print progress messages.
        """
        self.verbose = verbose

    def _log(self, message: str) -> None:
        """Log a message if verbose mode is enabled.
        
        Args:
            message: Message to log.
        """
        if self.verbose:
            print(f"[setup] {message}")

    def install_core_packages(self) -> None:
        """Install core packages for local environment."""
        self._log("Installing core packages...")
        
        core_packages = [
            "numpy>=1.22,<2.1",
            "scipy>=1.7.0,<1.14.0",
            "pandas>=2.2.2,<2.3",
            "scikit-learn>=1.5.2,<2.0",
        ]
        
        try:
            self._run_pip_command(["install", "--upgrade"] + core_packages)
        except subprocess.CalledProcessError as e:
            self._log(f"Warning: Failed to install core packages: {e}")

    def install_ml_packages(self) -> None:
        """Install ML-specific packages."""
        self._log("Installing ML-specific packages...")
        
        ml_packages = [
            "mamba-ssm>=2.2.5",
            "causal-conv1d>=1.5.2",
        ]
        
        try:
            self._run_pip_command(["install", "--upgrade"] + ml_packages)
        except subprocess.CalledProcessError as e:
            self._log(f"Warning: Failed to install ML packages: {e}")

    def install_from_requirements(self, requirements_path: Path) -> None:
        """Install all packages from requirements.txt.
        
        Args:
            requirements_path: Path to requirements.txt file.
        """
        if not requirements_path.exists():
            self._log(f"Requirements file not found: {requirements_path}")
            return

        self._log("Installing packages from requirements.txt...")
        
        try:
            self._run_pip_command(["install", "-r", str(requirements_path)])
            self._log("Requirements installed successfully")
        except subprocess.CalledProcessError as e:
            self._log(f"Warning: Failed to install requirements: {e}")


class InstallationContext:
    """Context class that uses an InstallationStrategy.
    
    This class delegates installation tasks to the configured strategy,
    decoupling the client code from specific installation implementations.
    """

    def __init__(self, strategy: InstallationStrategy) -> None:
        """Initialize installation context with a strategy.
        
        Args:
            strategy: The installation strategy to use.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: InstallationStrategy) -> None:
        """Change the installation strategy at runtime.
        
        Args:
            strategy: The new installation strategy to use.
        """
        self._strategy = strategy

    def install_all(self, requirements_path: Optional[Path] = None) -> None:
        """Execute complete installation process.
        
        Args:
            requirements_path: Optional path to requirements.txt file.
        """
        self._strategy.install_core_packages()
        self._strategy.install_ml_packages()
        
        if requirements_path:
            self._strategy.install_from_requirements(requirements_path)
