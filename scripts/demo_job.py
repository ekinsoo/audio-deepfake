"""Bootstrap the local demo with the actual Python executable and project venv."""

import os
import runpy

import full_run_job


if __name__ == "__main__":
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    full_run_job.configure_project_environment()
    runpy.run_module("audio_deepfake.demo", run_name="__main__")
