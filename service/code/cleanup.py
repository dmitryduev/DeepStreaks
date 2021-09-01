import argparse
import datetime
import pathlib
import shutil
import time


def time_stamp():
    """

    :return: UTC time -> string
    """
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H:%M:%S")


def log(message):
    """Print a log message with a UTC time stamp"""
    print(f"{time_stamp()}: {message}")


def main(
    date: datetime.datetime,
    path: pathlib.Path = pathlib.Path("/data/streaks"),
):
    """Remove stamps cached before <date>

    :param date:
    :param path:
    :return:
    """
    dirs = (path / "stamps").glob("stamps_*")
    dirs_to_remove = [
        str(directory) for directory in dirs
        if datetime.datetime.strptime(
            str(directory).split("_")[1],
            "%Y%m%d"
        ) < date
    ]

    for directory in dirs_to_remove:
        log(F"Removing {directory}")
        shutil.rmtree(directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean up ZTF nightly streak data"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=5,
        help="remove cached data older than n days"
    )
    parser.add_argument(
        "--run_once",
        action="store_true",
        help="run once and exit"
    )

    args = parser.parse_args()

    while True:
        today = datetime.datetime.utcnow()
        today = datetime.datetime(
            year=today.year,
            month=today.month,
            day=today.day
        )
        cutout_date = today - datetime.timedelta(days=int(args.days))

        main(cutout_date)
        if args.run_once:
            break
        # nap for a day:
        time.sleep(86400)
