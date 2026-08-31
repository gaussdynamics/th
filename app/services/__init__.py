"""Service layer — the only place that touches the backend / data files.

Views call services; services return plain data. No Qt in here, so each
service is unit-testable headless.
"""
