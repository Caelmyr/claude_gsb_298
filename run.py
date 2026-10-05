"""Start the web application."""

import argparse

from backend import seed
from backend.app import app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the scheduling web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--seed", action="store_true",
                        help="create example instances on startup")
    args = parser.parse_args()

    if args.seed:
        created = seed.seed_all()
        if created:
            print(f"seeded: {', '.join(created)}")

    print(f"serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
