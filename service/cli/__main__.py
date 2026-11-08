"""让 `python -m cli` 可用（当 service 在 sys.path 时）。"""
import sys

from cli.gargantua import main

sys.exit(main())
