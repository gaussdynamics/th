"""RailLab — desktop research console for longitudinal train dynamics.

Thin PySide6 UI layer over the existing simulator2 / routegen backend.
The UI never imports scipy or torch directly; it goes through app/services.
"""

__version__ = "0.1.0"
