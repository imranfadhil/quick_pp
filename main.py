import sys
from importlib import import_module


if __name__ == '__main__':
    main = getattr(import_module('quick_pp.cli'), 'cli')

    if (len(sys.argv) > 1):
        main(sys.argv[1:])
