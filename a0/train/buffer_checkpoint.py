'''
Replay-buffer checkpointing for resumable training (a0.train.alphazero).

Training resumes from the highest model_<i>.pkl, but historically the replay
buffer restarted empty on every resume (including at each auto-resubmit chain
boundary), so the first post-resume iteration trained only on freshly
generated games. When config.persist_replay_buffer is on, the buffer is
checkpointed to training_dir/buffer.pkl right after each model_<i>.pkl save
and restored on resume.

There is exactly ONE buffer.pkl per trial, overwritten each iteration — only
the state paired with the newest model checkpoint is kept, because that is the
only state resume can use. The file is tagged with its iteration; on load a
tag that doesn't match the checkpoint being resumed from is refused, and
training falls back to the pre-existing empty-buffer resume behavior.

A crash between the model_<i+1> save and the buffer save (or mid buffer
write, which the atomic tmp+rename turns into the same situation) leaves the
buffer exactly one iteration behind the newest model. Rather than resuming
from model_<i+1> with an empty buffer, rollback_for_buffer_consistency
deletes the unpaired model_<i+1> and resumes from the consistent
(model_<i>, buffer_<i>) pair — redoing one iteration instead of refilling
the buffer from scratch. Any other gap is NOT rolled back (it means
something other than a mid-save crash, e.g. persistence enabled partway
through a run) and resume proceeds from the newest model, empty-buffered.

Only plain state is pickled (numpy arrays + python scalars):
  - ExperienceBuffer: the list of ExperienceData samples.
  - TrajectoryReplayBuffer: the list of TrajectoryData games. Safe at the
    end-of-iteration save point because game_data refs have been dropped and
    the sidecar popped by then; the C++-backed ranker is never pickled — the
    ctor-built one is kept on load.
'''
from collections import deque
import os
import pickle

from config import config

from a0.experience_buffer import ExperienceBuffer
from a0.train.trajectory_buffer import TrajectoryReplayBuffer

from utils.log import get_logger
logger = get_logger(__name__)

def replay_buffer_path() -> str:
    return config.training_dir + "buffer.pkl"

def _read_payload() -> dict | None:
    '''The buffer checkpoint's payload dict, or None (with a log line) if the
    file is missing or unreadable.'''
    path = replay_buffer_path()
    if not os.path.exists(path):
        logger.info("No replay buffer checkpoint found; starting with an empty buffer.")
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.log(30, f"Failed to load replay buffer checkpoint {path} ({e}); starting empty.")
        return None

def rollback_for_buffer_consistency(model_path: str, starting_iteration: int) -> tuple[str, int]:
    '''
    Given the newest model checkpoint resume would use, decide whether to roll
    back one iteration to pair with the buffer checkpoint. Rolls back ONLY
    when buffer.pkl is tagged exactly starting_iteration - 1 (the signature of
    a crash between/during the end-of-iteration saves) and model_<i-1>.pkl
    exists: the unpaired newest model is deleted and the paired (model, buffer)
    iteration is returned. Otherwise returns the input unchanged.
    '''
    payload = _read_payload()
    if payload is None:
        return model_path, starting_iteration
    tag = payload.get("iteration")
    if tag != starting_iteration - 1:
        return model_path, starting_iteration
    prev_path = config.training_dir + f"model_{tag}.pkl"
    if not os.path.exists(prev_path):
        logger.log(30, f"Buffer checkpoint is from iteration {tag} but {prev_path} is missing; "
                       f"resuming from iteration {starting_iteration} with an empty buffer.")
        return model_path, starting_iteration
    os.remove(model_path)
    logger.log(25, f"Rolled back to iteration {tag}: deleted unpaired {model_path} so training "
                   f"resumes with the matching replay buffer instead of an empty one.")
    return prev_path, tag

def save_replay_buffer(iteration: int, buffer: ExperienceBuffer | TrajectoryReplayBuffer) -> None:
    '''
    Checkpoint the replay buffer alongside model_<iteration>.pkl, overwriting
    the previous iteration's buffer.pkl. Written atomically (tmp + rename) so
    a crash mid-write can't corrupt the previous buffer checkpoint.
    '''
    if isinstance(buffer, TrajectoryReplayBuffer):
        payload = {"iteration": iteration, "kind": "trajectory",
                   "state": list(buffer.games)}
    else:
        payload = {"iteration": iteration, "kind": "experience",
                   "state": list(buffer.data)}
    path = replay_buffer_path()
    tmp_path = path + ".tmp"
    with open(tmp_path, 'wb') as f:
        pickle.dump(payload, f)
    os.replace(tmp_path, path)

def load_replay_buffer(starting_iteration: int, buffer: ExperienceBuffer | TrajectoryReplayBuffer) -> bool:
    '''
    Restore the replay buffer saved next to model_<starting_iteration>.pkl.
    Returns True on success. If the buffer file is missing, unreadable, from a
    different iteration (e.g. its paired checkpoint was deleted), or of the
    wrong kind, logs why and leaves the buffer empty — the pre-existing resume
    behavior.
    '''
    payload = _read_payload()
    if payload is None:
        return False
    expected_kind = "trajectory" if isinstance(buffer, TrajectoryReplayBuffer) else "experience"
    if payload.get("kind") != expected_kind:
        logger.log(30, f"Replay buffer checkpoint kind {payload.get('kind')!r} != {expected_kind!r}; starting empty.")
        return False
    if payload.get("iteration") != starting_iteration:
        logger.log(30, f"Replay buffer checkpoint is from iteration {payload.get('iteration')}, "
                       f"resuming from {starting_iteration}; starting empty.")
        return False
    if isinstance(buffer, TrajectoryReplayBuffer):
        buffer.games = deque(payload["state"])
        buffer.num_positions = sum(len(traj) for traj in buffer.games)
        logger.info(f"Restored trajectory buffer: {buffer.num_positions} positions "
                    f"in {len(buffer.games)} games (iteration {starting_iteration}).")
    else:
        buffer.data.extend(payload["state"])
        logger.info(f"Restored experience buffer: {len(buffer)} positions (iteration {starting_iteration}).")
    return True
