import importlib
import sys

from absl import app

AVAILABLE_SCRIPTS = ['preprocess', 'train', 'export', 'compact', 'combine']


def help():
    print(f"""usage: msprior [ {' | '.join(AVAILABLE_SCRIPTS)} ]

positional arguments:
  command     Command to launch with msprior.
""")
    exit()


def main():
    if len(sys.argv) == 1:
        help()
    elif sys.argv[1] not in AVAILABLE_SCRIPTS:
        help()

    command = sys.argv[1]

    module = importlib.import_module("msprior_scripts." + command)
    sys.argv[0] = module.__name__
    app.run(module.main)
