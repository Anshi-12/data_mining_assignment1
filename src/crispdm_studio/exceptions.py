"""Domain exceptions with safe user-facing messages."""


class CrispDMError(Exception):
    """Base application exception intended to be safely presented in the UI."""

    default_message = "We couldn't complete that operation."

    def __init__(self, message: str | None = None, *, detail: str | None = None) -> None:
        super().__init__(message or self.default_message)
        self.user_message = message or self.default_message
        self.detail = detail


class UploadValidationError(CrispDMError):
    default_message = "We couldn't validate the uploaded file."


class UnsupportedFileTypeError(UploadValidationError):
    default_message = "Please upload a CSV file."


class EmptyFileError(UploadValidationError):
    default_message = "We couldn't analyze this file because it is empty."


class EncodingDetectionError(UploadValidationError):
    default_message = "We couldn't determine the text encoding of this CSV."


class DelimiterDetectionError(UploadValidationError):
    default_message = "We couldn't determine how the CSV columns are separated."


class CSVReadError(UploadValidationError):
    default_message = "We couldn't read this CSV. Please check that it is a valid delimited text file."


class ZeroRowError(UploadValidationError):
    default_message = "We couldn't analyze this file because it contains column headers but no data rows."


class NoColumnsError(UploadValidationError):
    default_message = "We couldn't analyze this file because no columns were detected."


class DimensionLimitError(UploadValidationError):
    default_message = "This dataset is larger than the configured analysis limits."


class HeaderValidationError(UploadValidationError):
    default_message = "We couldn't analyze this file because its column headers are unusable."
