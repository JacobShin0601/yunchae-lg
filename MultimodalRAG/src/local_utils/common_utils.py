import time
import pickle
import random
import logging
import functools
from IPython.display import Markdown, HTML, display
from io import StringIO
import sys
import textwrap
import json
import base64

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def print_ww(*args, width: int = 100, **kwargs):
    """Like print(), but wraps output to `width` characters (default 100)"""
    buffer = StringIO()
    try:
        _stdout = sys.stdout
        sys.stdout = buffer
        print(*args, **kwargs)
        output = buffer.getvalue()
    finally:
        sys.stdout = _stdout
    for line in output.splitlines():
        print("\n".join(textwrap.wrap(line, width=width)))


# 재수행 데코레이터
def retry(total_try_cnt=10, sleep_in_sec=20, retryable_exceptions=()):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, total_try_cnt + 1):
                logger.info(
                    f"Trying {func.__name__}(): attempt {attempt}/{total_try_cnt}"
                )

                try:
                    result = func(*args, **kwargs)
                    logger.info(f"{func.__name__}() returned: {result}")

                    # 데이터 처리 결과가 None이 아닌 경우 반환
                    if result is not None:
                        return result
                except retryable_exceptions as e:
                    logger.info(
                        f"Retryable error in {func.__name__}(): {e}. Retrying..."
                    )
                except Exception as e:
                    logger.error(
                        f"Non-retryable error in {func.__name__}(): {e}. Aborting..."
                    )
                    raise

                time.sleep(sleep_in_sec)
            logger.error(f"{func.__name__}() failed after {total_try_cnt} attempts")

        return wrapper

    return decorator


def to_pickle(obj, path):
    with open(file=path, mode="wb") as f:
        pickle.dump(obj, f)

    print(f"to_pickle file: {path}")


def load_pickle(path):
    with open(file=path, mode="rb") as f:
        obj = pickle.load(f)

    print(f"load from {path}")

    return obj


def load_json(file_path):
    """
    Load JSON data from a file.
    """
    with open(file_path, "r") as f:
        return json.load(f)


def to_json(obj, file_path):
    """
    Save a Python object as a JSON file.
    """
    with open(file_path, "w") as f:
        json.dump(obj, f, indent=4)


def to_markdown(obj, path):
    with open(file=path, mode="w") as f:
        f.write(obj)
    print(f"mark down: {path}")


def print_html(input_html):
    html_string = " "
    html_string = html_string + input_html
    display(HTML(html_string))


def image_to_base64(image_path):
    """Convert an image file to a Base64 encoded string."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string.decode("utf-8")
