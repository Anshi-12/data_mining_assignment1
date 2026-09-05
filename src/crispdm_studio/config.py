"""Application-wide configuration and safety limits."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AppConfig:
    app_name: str = "CRISP-DM Studio"
    max_upload_mb: int = 50
    max_rows: int = 500_000
    max_columns: int = 500
    preview_rows: int = 100
    encoding_sample_bytes: int = 256_000
    delimiter_sample_chars: int = 64_000
    profile_max_cells: int = 25_000_000
    eda_max_rows: int = 100_000
    clustering_max_rows: int = 20_000
    modeling_max_rows: int = 50_000
    permutation_max_rows: int = 5_000
    permutation_repeats: int = 5
    allowed_extensions: tuple[str, ...] = (".csv",)
    allowed_mime_types: tuple[str, ...] = (
        "text/csv",
        "application/csv",
        "application/vnd.ms-excel",
        "text/plain",
        "application/octet-stream",
        "",
    )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


CONFIG = AppConfig()
