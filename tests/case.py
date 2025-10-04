import json
from pathlib import Path
from typing import Any

from nunchaku.utils import get_precision


class Case:
    def __init__(
        self,
        workflow_name: str,
        ref_image_url: str,
        expected_clip_iqa: dict[str, float] = {},
        expected_lpips: dict[str, float] = {},
        expected_psnr: dict[str, float] = {},
        inputs: dict[tuple, Any] = {},
    ):
        self.workflow_name = workflow_name
        self.ref_image_url = ref_image_url
        self.expected_clip_iqa = expected_clip_iqa
        self.expected_lpips = expected_lpips
        self.expected_psnr = expected_psnr
        self.inputs = inputs


def collect_cases() -> tuple[list[Case], list[str]]:
    ret_cases = []
    ret_ids = []
    # Find all test_cases.json files in workflow folders under the same directory
    current_dir = Path(__file__).parent / "workflows"
    dirs = sorted([d for d in current_dir.iterdir() if d.is_dir()])
    for workflow_dir in dirs:
        test_cases_path = workflow_dir / "test_cases.json"
        if test_cases_path.exists():
            with open(test_cases_path, "r", encoding="utf-8") as f:
                test_cases = json.load(f)
                for id, case_data in enumerate(test_cases):
                    test_case = Case(workflow_name=workflow_dir.name, **case_data)
                    for k in test_case.inputs.keys():
                        if isinstance(test_case.inputs[k], str):
                            test_case.inputs[k] = test_case.inputs[k].format(precision=get_precision())
                    ret_cases.append(test_case)
                    ret_ids.append(workflow_dir.name + "_" + str(id))
    return ret_cases, ret_ids


cases, ids = collect_cases()
