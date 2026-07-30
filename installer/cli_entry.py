"""Entry point for the console executable.

The UTF-8 console handling lives in `cli.main()`, so the packaged and unpackaged
CLI behave identically and there is only one place to fix if it ever regresses.
"""

from svrspec.cli import main

raise SystemExit(main())
