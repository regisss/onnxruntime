#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import subprocess
import sys


def run_lazy_tensor_tests():
    #  Try debug flags to print more info.
    env = {
        "ORT_LT_DUMP_GRAPH": "1",
        "ORT_LT_DUMP_INPUTS_OUTPUTS": "1",
        "ORT_LT_CHECK_BASELINE": "1",
        "ORT_LT_DUMP_ATEN_OP_HISTORY": "1",
    }

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-sv",
        "/bert_ort/wechi/ltc/ort/orttraining/orttraining/test/python/orttraining_test_lort.py",
    ]

    subprocess.run(command, env=env).check_returncode()


def main():
    run_lazy_tensor_tests()
    return 0


if __name__ == "__main__":
    sys.exit(main())
