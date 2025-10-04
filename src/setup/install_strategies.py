"""Installation strategies for different environments using Strategy Pattern.

This module implements the Strategy Pattern to handle package installation
across different environments (Kaggle, local, etc.) with clean separation
of concerns and extensibility.
"""

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
        capture_output: bool = True
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
        return subprocess.run(cmd, check=check, capture_output=capture_output)

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
                # Extract version from pip show output
                output = result.stdout.decode("utf-8")
                for line in output.split("\n"):
                    if line.startswith("Version:"):
                        installed_version = line.split(":")[1].strip()
                        # Simple version comparison (works for most cases)
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
        """Install core packages with Kaggle-specific handling."""
        self._log("Checking core packages...")
        
        core_packages = [
            ("numpy", "1.22"),
            ("scipy", "1.7.0"),
            ("pandas", "2.2.2"),
            ("scikit-learn", "1.5.2"),
        ]
        
        packages_to_install = []
        for pkg_name, min_version in core_packages:
            if not self._is_package_installed(pkg_name, min_version):
                self._log(f"{pkg_name} needs installation/upgrade")
                packages_to_install.append(pkg_name)
            else:
                self._log(f"✓ {pkg_name} already installed with compatible version")
        
        if not packages_to_install:
            self._log("All core packages are up to date")
            return
        
        # Install only packages that need it
        package_specs = {
            "numpy": "numpy>=1.22,<2.1",
            "scipy": "scipy>=1.7.0,<1.14.0",
            "pandas": "pandas>=2.2.2,<2.3",
            "scikit-learn": "scikit-learn>=1.5.2,<2.0",
        }
        
        for pkg_name in packages_to_install:
            try:
                self._log(f"Installing {package_specs[pkg_name]}...")
                self._run_pip_command(
                    ["install", "--upgrade", package_specs[pkg_name]],
                    capture_output=True
                )
                self._log(f"✓ {pkg_name} installed successfully")
            except subprocess.CalledProcessError as e:
                self._log(f"Warning: Failed to install {pkg_name}: {e}")

    def install_ml_packages(self) -> None:
        """Install ML-specific packages like mamba-ssm.
        
        Builds mamba-ssm from source using GPU for compilation.
        This ensures compatibility with the specific environment.
        """
        self._log("Checking ML-specific packages...")
        
        # Check if mamba-ssm is already installed
        if self._is_package_installed("mamba-ssm"):
            self._log("✓ mamba-ssm already installed")
        else:
            try:
                self._log("Building mamba-ssm from source (GPU-accelerated)...")
                # Build from source for environment-specific optimization
                # GPU will be used automatically during CUDA compilation
                result = self._run_pip_command(
                    ["install", "git+https://github.com/state-spaces/mamba.git"],
                    capture_output=False
                )
                self._log("✓ mamba-ssm built and installed successfully")
            except subprocess.CalledProcessError as e:
                self._log(f"Warning: Failed to build mamba-ssm: {e}")
        
        # Check causal-conv1d
        if self._is_package_installed("causal-conv1d"):
            self._log("✓ causal-conv1d already installed")
        else:
            try:
                self._log("Installing causal-conv1d...")
                result = self._run_pip_command(
                    ["install", "causal-conv1d>=1.5.2"],
                    capture_output=False
                )
                self._log("✓ causal-conv1d installed successfully")
            except subprocess.CalledProcessError as e:
                self._log(f"Warning: Failed to install causal-conv1d: {e}")

    def install_from_requirements(self, requirements_path: Path) -> None:
        """Install remaining packages from requirements.txt.
        
        Args:
            requirements_path: Path to requirements.txt file.
        """
        if not requirements_path.exists():
            self._log(f"Requirements file not found: {requirements_path}")
            return

        self._log("Installing remaining packages from requirements.txt...")
        
        # Packages to skip (already handled in install_core_packages and install_ml_packages)
        skip_packages = {
            "torch", "numpy", "scipy", "pandas", "scikit-learn"
        }
        
        # Create filtered requirements file
        filtered_path = Path("/tmp/requirements_filtered.txt")
        with open(requirements_path, "r") as rf, open(filtered_path, "w") as wf:
            for line in rf:
                pkg = line.strip()
                if not pkg or pkg.startswith("#"):
                    continue
                
                # Skip git+ URLs (mamba-ssm) - handled in install_ml_packages
                if pkg.startswith("git+"):
                    continue
                
                # Extract package name (handle various formats)
                pkg_name = (
                    pkg.split("[")[0]
                    .split("==")[0]
                    .split(">=")[0]
                    .split("<")[0]
                    .strip()
                    .lower()
                )
                
                if pkg_name not in skip_packages:
                    wf.write(line)

        try:
            self._run_pip_command(
                ["install", "-r", str(filtered_path), "--quiet"]
            )
            self._log("Requirements installed successfully")
        except subprocess.CalledProcessError as e:
            self._log(f"Warning: Some requirements failed to install: {e}")


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
