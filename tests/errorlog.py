import datetime


def log_error(error_message: str, source: str = ""):
    """Logs error messages to error_log.txt with a timestamp and optional source."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("error_log.txt", "a") as log_file:
        log_file.write(f"[{timestamp}] {source}: {error_message}\n")