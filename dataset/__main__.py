"""Entry point for dataset creation module."""

try:
    from .create_multi_asset_dataset import main
except ImportError:
    from create_multi_asset_dataset import main

if __name__ == '__main__':
    main()
