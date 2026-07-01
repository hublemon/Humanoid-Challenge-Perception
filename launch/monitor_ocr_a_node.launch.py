import importlib.util
from pathlib import Path


_source_path = Path(__file__).with_name('monitor_ocr_a.launch.py')
_spec = importlib.util.spec_from_file_location('monitor_ocr_a_launch', _source_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

generate_launch_description = _module.generate_launch_description
