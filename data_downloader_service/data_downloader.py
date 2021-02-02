import argparse
import os
import re
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Data Downloader')
    parser.add_argument('--data-url', type=str, required=True,
                        help='URL to file with data to download.')
    parser.add_argument('--save-loc', type=str, required=True,
                        help='Path to file where downloaded data will be saved.')
    return parser.parse_args()


def assert_url_validity(url: str) -> None:
    """
    Checks whether given link is valid.
    """
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    assert re.match(regex, url) is not None


if __name__ == '__main__':
    args: argparse.Namespace = parse_args()

    assert_url_validity(args.data_url)
    os.makedirs(Path(args.save_loc).parent, exist_ok=True)

    print('Beginning file download with urllib2...')

    urllib.request.urlretrieve(args.data_url, args.save_loc)

    print(f'Done! File saved in {args.save_loc}')
