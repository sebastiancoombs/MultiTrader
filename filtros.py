import sys
from typing import TextIO, Iterable
import logging
import re
import warnings

class FiltrosStream:
    def __init__(self, original_stream: TextIO, patterns: Iterable[str] = ()) -> None:
        self.original_stream = original_stream
        self.patterns = tuple(patterns)
        self.combined_pattern = re.compile('|'.join(f'(?:{pattern})' for pattern in patterns))
        self.last_printed = True

    def write(self, message: str) -> None:
        # print(f'XXX: {repr(message)}')
        if message.isspace() and not self.last_printed:
            return
        if self.combined_pattern.search(message):
            self.last_printed = False
            return
        else:
            self.last_printed = True
            self.original_stream.write(message)

    def flush(self) -> None:
        self.original_stream.flush()

    def fileno(self) -> int:
        return self.original_stream.fileno()

    def close(self) -> None:
        pass


def filtros(patterns) -> None:
    sys.stderr = FiltrosStream(sys.stderr, patterns)


def activate() -> None:
    # Commented out because it doesn't work for all cases:
    # warnings.filterwarnings('ignore', 'The distutils package is deprecated',
                            # DeprecationWarning)
    filtros(
        (
            r'The distutils package is deprecated',
            r'import distutils',

            r"Setting 'object_store_memory' for actors",

            r'DeprecationWarning: non-integer arguments to randrange\(\)',
            r'self.episode_id: int = random\.randrange\(2e9\)',

            r'Install gputil for GPU system monitoring',

            r'ray/rllib.* DeprecationWarning: `np\.bool` is a deprecated alias for the',
            r'Deprecated in NumPy 1\.20; for more details and guidance',
            r'if not isinstance\(done_, \(bool, np\.bool, np\.bool_\)\):',

            r'/ray/dashboard.* DeprecationWarning: There is no current event loop',
            r'aiogrpc\.init_grpc_aio\(\)',
            r'loop = asyncio\.get_event_loop\(\)',

            r'(?:observation|action)_space_(?:contains|sample)\(\) has not been implemented',

            r'DeprecationWarning: `_get_slice_indices` has been deprecated',

            r'Current log_level is .{1,20}For more information,',

            r'[0-9]{2},[0-9]{3}\s+INFO ',

            r'Function `rng.randint\(low, \[high, size, dtype\]\)` is marked as deprecated',

            r'past/.*DeprecationWarning: the imp module',
            r'from imp import reload',

            r'DeprecationWarning: `tune.report` and `tune.checkpoint_dir` APIs are deprecated',

            r'train_batch_size.*cannot be achieved',

            r'Function checkpointing is disabled\. This may result',

            (r'DeprecationWarning: `(DirectStepOptimizer|build_policy_class|build_tf_policy|'
             r'ValueNetworkMixin|LearningRateSchedule|EntropyCoeffSchedule|KLCoeffMixin)` has '
             r'been deprecated'),

            r'Setting `exploration_config={}` because you set `_enable_rl_module_api=True`',

            r'ray/rllib/algorithms/algorithm.py:484: RayDeprecationWarning: This API is deprecated',
            r'`UnifiedLogger` will be removed',

            r'ray/tune/logger/unified.py:53: RayDeprecationWarning: This API is deprecated',
            r'The `[a-zA-Z0-9]{1,40}Logger`? interface is deprecated',
            r'self._loggers.append(cls(self.config, self.logdir, self.trial))',


        )
    )
