import argparse

from .pipeline import run_from_file


def main():
    parser = argparse.ArgumentParser(prog="qmldd", description="Hybrid QML disease detection platform")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run a benchmark experiment from a YAML config")
    run_parser.add_argument("--config", required=True, help="Path to a config YAML file")

    args = parser.parse_args()

    if args.command == "run":
        run_from_file(args.config)


if __name__ == "__main__":
    main()
