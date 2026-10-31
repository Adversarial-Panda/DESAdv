import os
from pathlib import Path

# Halligan's model weights live in <project>/weights/halligan (see the project README.md).
WEIGHTS_DIR = Path(os.environ.get('HALLIGAN_WEIGHTS',
                                  Path(__file__).resolve().parents[3] / 'weights' / 'halligan'))
HF_CACHE = str(WEIGHTS_DIR / 'huggingface')


class Model(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        """All models are singletons that can only be instantiated once throughout the system
        """
        if cls not in cls._instances:
            cls._instances[cls] = super(Model, cls).__call__(*args, **kwargs)
        return cls._instances[cls]