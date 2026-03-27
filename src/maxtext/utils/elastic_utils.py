# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for Elastic Training."""

import functools
import jax
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging
import pathwaysutils
from pathwaysutils.elastic import manager


elastic_manager: manager.Manager | None = None


def elastic_mode_enabled(config) -> bool:
  """Returns whether elastic mode is enabled."""
  return pathwaysutils.is_pathways_backend_used() and config.elastic_pause_resume


def clean_up_checkpoints(checkpoint_dir: str):
  """Cleans up incomplete checkpoints after an elastic event."""
  max_logging.log("Elastic utils: Checking for incomplete checkpoint after an elastic event...")
  checkpoint_dir = gcs_utils.add_trailing_slash(checkpoint_dir)

  # 1. List the "directories" (steps)
  checkpoints = gcs_utils.gcs_list_directories(checkpoint_dir)

  # 2. Filter for directories that are numbers
  checkpoints = [cp for cp in checkpoints if cp.isdigit()]

  if not checkpoints:
    max_logging.log("Found no existing checkpoints. Continuing")
    return

  # Sort naturally (numerical sort) and get the last one
  checkpoints.sort(key=int)
  latest_checkpoint_name = checkpoints[-1]
  latest_checkpoint_path = f"{checkpoint_dir}{latest_checkpoint_name}/"

  max_logging.log(f"Checking latest checkpoint: {latest_checkpoint_path}")

  # 3. Check for commit_success file
  success_markers = gcs_utils.gcs_glob_pattern(f"{latest_checkpoint_path}commit_success*")

  if not success_markers:
    max_logging.log(f"No commit_success file found. Deleting {latest_checkpoint_path}...")
    gcs_utils.gcs_delete_directory(latest_checkpoint_path)
  else:
    max_logging.log(f"Found commit_success file. Keeping {latest_checkpoint_path}.")


def live_devices():
  """Returns the list of live devices."""
  global elastic_manager
  # If pathways is not used or elastic_manager is not initialized, return all devices
  if pathwaysutils.is_pathways_backend_used():
    if elastic_manager is None:
      elastic_manager = manager.Manager()
    # Filter devices that are in active slices
    return [d for d in jax.devices() if d.slice_index in elastic_manager.active_slice_indices]
  return jax.devices()


def elastic_pause_resume(config, callback_fn=None):
  """Decorator for elastic pause/resume."""
  global elastic_manager
  if not elastic_mode_enabled(config):
    return lambda x: x

  max_logging.log("Elastic Pause and Resume Enabled")
  if elastic_manager is None:
    elastic_manager = manager.Manager()

  cleanup_partial = functools.partial(clean_up_checkpoints, config.checkpoint_dir)
  callback_fn = cleanup_partial if callback_fn is None else callback_fn

  return elastic_manager.elastic_retry(
      max_retries=config.elastic_max_retries,
      poll_interval=config.elastic_retry_interval_seconds,
      timeout=config.elastic_timeout_seconds,
      on_elastic_event_callback=callback_fn,
  )
