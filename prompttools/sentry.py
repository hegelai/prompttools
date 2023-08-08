# Sentry collects crash reports and performance numbers
# It is possible to turn off data collection using an environment variable named "SENTRY_OPT_OUT"
import sentry_sdk

import os
import platform
import uuid
import hashlib
from .version import __version__


SENTRY_DSN = "https://43fbb5a3a556ca0a879f5a08ce805d87@o4505656408211456.ingest.sentry.io/4505656412667904"

# Get a random token based on the machine uuid
token = hashlib.sha256(str(uuid.getnode()).encode()).hexdigest()


def find_certifi_path():
    try:
        import certifi

        return os.path.join(os.path.dirname(certifi.__file__), "cacert.pem")
    except Exception:
        pass
    return None


def filter_info(event, _hint):
    # Remove personal info
    try:
        event["modules"] = None
        event["extra"] = None
        event["server_name"] = None
    except Exception:
        pass
    return event


def init_sentry():
    if "SENTRY_OPT_OUT" not in os.environ:
        if platform.system() == "Darwin":
            # Fix CA certificate issue on latest MAC models
            path = find_certifi_path()
            if path is not None:
                if "SSL_CERT_FILE" not in os.environ:
                    os.environ["SSL_CERT_FILE"] = path
                if "REQUESTS_CA_BUNDLE" not in os.environ:
                    os.environ["REQUESTS_CA_BUNDLE"] = path

        sentry_sdk.init(
            dsn=SENTRY_DSN,
            release=__version__,
            traces_sample_rate=1,
            include_local_variables=False,
            send_default_pii=False,
            attach_stacktrace=False,
            before_send=filter_info,
            include_source_context=False,
        )
        try:
            filename = os.path.join(os.environ.get("HOME", "/tmp"), ".token")
            if platform.system() == "Windows":
                filename = os.path.join(os.environ.get("USERPROFILE", "c:\\"), ".token")
            with open(filename, "w") as f:
                f.write(token)
        except Exception:
            pass

    sentry_sdk.capture_message("Initializing prompttools", "info")
