import inspect
import importlib
from hydra.core.config_store import ConfigStore
from pathlib import Path
from typing import Any, Type, Union, Optional
from types import ModuleType
import pkgutil
from dataclasses import is_dataclass


def auto_register(base_class: Type, config_root: Optional[Union[str, Path]]):
    """Auto register config that inherits a base class.
    
    This automatically registers all the config class defined in the same package
    as baseclass. Rules:
    - The base class must be defined in the __init__.py of the package
    - Subclasses must be defined in direct modules of that package
    - Each module file should only contain one subclass.
    """
    config_root = config_root
    assert config_root.stem == "configs", "Just a sanity check, you can change this"

    pkg = importlib.import_module(base_class.__module__)
    assert hasattr(pkg, "__path__"), (
        f"{base_class}'s module does not have attribute __path__. {base_class}"
        f" must be defined in `__init__.py` in order for auto register to work"
    )
    pkg_path = Path(pkg.__file__).parent

    try:
        group = str(pkg_path.relative_to(config_root))
    except ValueError:
        raise ValueError(
            f"Node {pkg.__name__}'s path {pkg_path} is not under config root {config_root}."
        )

    cs = ConfigStore.instance()
    for loader, module_name, is_pkg in pkgutil.iter_modules(pkg.__path__):
        module = importlib.import_module(f"{pkg.__name__}.{module_name}")
        # Iterate through the attributes of the module
        valid_list = []
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, base_class)
                and obj is not base_class
            ):
                assert is_dataclass(obj), f"{obj} must be dataclass"
                valid_list.append((name, obj))
        if len(valid_list) != 1:
            raise ValueError(
                f"Module {module} should define exactly one subclass of {base_class}, but got {valid_list}"
            )
        else:
            name, obj = valid_list[0]
            cs.store(name=module_name, group=group, node=obj)
