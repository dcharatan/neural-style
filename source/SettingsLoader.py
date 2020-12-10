import json
from argparse import ArgumentParser
from typing import Optional


class SettingsLoader:
    SETTINGS_DEFAULTS = {
        "image_size": 256,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "num_epochs": 1,
        "feature_weight": 1,
        "style_weight": 20000,
        "results_folder": "results/mosaic",
        "style_image": "style/mosaic.jpg",
        "save_model_every": 1000,
        "save_image_every": 25,
        "print_status_every": 50,
        "per_channel_normalize": False,
    }

    @staticmethod
    def load_settings_from_argv():
        parser = ArgumentParser(description="Script with JSON Settings File")
        parser.add_argument("--file_name", help="The JSON settings file.")
        args = parser.parse_args()
        return SettingsLoader.load_settings(args.file_name)

    @staticmethod
    def load_settings(file_name: Optional[str]):
        if file_name is None:
            return SettingsLoader.SETTINGS_DEFAULTS

        with open(file_name, "r") as f:
            settings = json.load(f)
        SettingsLoader.validate_settings(settings)

        # Use the defaults for unspecified settings.
        return {**SettingsLoader.SETTINGS_DEFAULTS, **settings}

    @staticmethod
    def validate_settings(settings: dict):
        for key in settings:
            if key not in SettingsLoader.SETTINGS_DEFAULTS:
                raise Exception(f'Unknown setting "{key}"')
