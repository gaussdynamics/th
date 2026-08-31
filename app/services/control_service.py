"""Control-profile library adapter for the Controls screen.

Thin wrapper over ``simulator2.control_profile`` -- no Qt in here, easy to
unit test.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from simulator2.control_profile import (
    ControlProfile,
    DEFAULT_CONTROL_PROFILES_DIR,
    RandomProfileConfig,
    generate_random_profile,
    list_saved_control_profiles,
)


class ControlService:
    def __init__(self, directory: Path | str = DEFAULT_CONTROL_PROFILES_DIR) -> None:
        self.directory = Path(directory)

    def list_saved(self) -> List[Path]:
        return list_saved_control_profiles(self.directory)

    def load(self, path: Path | str) -> ControlProfile:
        return ControlProfile.load(path)

    def save(self, profile: ControlProfile, filename: Optional[str] = None) -> Path:
        return profile.save(self.directory, filename)

    def delete(self, path: Path | str) -> None:
        Path(path).unlink(missing_ok=True)

    def duplicate(self, path: Path | str, new_name: str) -> Path:
        profile = self.load(path)
        profile.name = new_name
        return self.save(profile)

    def rename(self, path: Path | str, new_name: str) -> Path:
        old_path = Path(path)
        profile = self.load(old_path)
        profile.name = new_name
        new_path = self.save(profile)
        if new_path != old_path:
            self.delete(old_path)
        return new_path

    def generate_random(self, config: RandomProfileConfig) -> ControlProfile:
        return generate_random_profile(config)
