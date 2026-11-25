import os

from config.load import load_config_file

# Environment variables that override the translation settings in config.yaml, so keys
# can be passed to a container (docker run -e GEMINI_API_KEY=...) instead of being
# written into a file: provider name -> key variable.
API_KEY_ENV = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}
# Which provider to try first ("gemini", "groq" or "deepseek").
TRANSLATOR_ENV = "NEURALINGO_TRANSLATOR"


def _apply_env_overrides(api_config: dict) -> dict:
    api_config = dict(api_config)
    for provider, variable in API_KEY_ENV.items():
        if os.environ.get(variable):
            api_config[provider] = {**(api_config.get(provider) or {}), "api_key": os.environ[variable]}
    if os.environ.get(TRANSLATOR_ENV):
        api_config["selected"] = os.environ[TRANSLATOR_ENV]
    return api_config


class ConfigBase:
    def __init__(self):
        config = load_config_file()

        self.config_api = _apply_env_overrides(config.get("api_group") or {})
        self.config_detection = config.get("detection_group", {})
        self.config_cleaning = config.get("cleaning_group", {})
        self.config_translation = config.get("translation_group", {})
        self.config_recognition = config.get("recognition_group", {})


    @property
    def api_config(self):
        return self.config_api


    @property
    def detection_config(self):
        return self.config_detection


    @property
    def cleaning_config(self):
        return self.config_cleaning


    @property
    def translation_config(self):
        return self.config_translation

    @property
    def recognition_config(self):
        return self.config_recognition