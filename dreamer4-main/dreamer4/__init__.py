try:
    from dreamer4.dreamer4 import (
        VideoTokenizer,
        DynamicsWorldModel,
        AxialSpaceTimeTransformer,
    )
except ImportError:  # pragma: no cover - optional dependencies for full package import
    VideoTokenizer = None
    DynamicsWorldModel = None
    AxialSpaceTimeTransformer = None

try:
    from dreamer4.kmc import (
        KMCGraphEncoder,
        KMCDynamicsWorldModel,
    )
except ImportError:  # pragma: no cover - optional dependencies for KMC extensions
    KMCGraphEncoder = None
    KMCDynamicsWorldModel = None

try:
    from dreamer4.trainers import (
        VideoTokenizerTrainer,
        BehaviorCloneTrainer,
        DreamTrainer,
    )
except ImportError:  # pragma: no cover - optional dependencies for trainer stack
    VideoTokenizerTrainer = None
    BehaviorCloneTrainer = None
    DreamTrainer = None

