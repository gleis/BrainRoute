"""Include the single source config catalog in built distributions."""
from pathlib import Path
from shutil import copy2
from setuptools import setup
from setuptools.command.build_py import build_py

class BuildWithDefaults(build_py):
    def run(self):
        super().run()
        destination = Path(self.build_lib) / "modelroute" / "defaults"
        destination.mkdir(parents=True, exist_ok=True)
        for source in (Path(__file__).parent / "config").glob("*.yaml"):
            copy2(source, destination / source.name)

setup(cmdclass={"build_py": BuildWithDefaults})
