from switch_trust.cli.main import main

# `raise SystemExit(main())`, not a bare `main()`: the installed console script
# is generated as `sys.exit(main())`, so dropping the return value here would
# make `python -m switch_trust.cli` (and the Bazel binary, which runs this same
# file) exit 0 on a run the installed binary fails. See exit_codes.py.
raise SystemExit(main())
