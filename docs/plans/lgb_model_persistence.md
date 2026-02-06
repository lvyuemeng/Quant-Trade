# Minimal Model Persistence Design

## Core Components

```python
# src/quant_trade/persistence.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, TypeVar
import pickle
from pathlib import Path

T = TypeVar('T')

@dataclass
class ModelContainer[T]:
    """Generic container for model + metadata"""
    model: T
    meta: 'ModelMeta'

@dataclass
class ModelMeta:
    """Immutable model metadata"""
    name: str
    created_at: datetime = field(default_factory=datetime.now)
    feature_names: list[str] = field(default_factory=list)
    metric_name: str | None = None
    metric_value: float | None = None
    tags: list[str] = field(default_factory=list)

class Storage(Protocol):
    """Storage protocol interface"""
    def save(self, name: str, container: ModelContainer[Any]) -> None: ...
    def load(self, name: str) -> ModelContainer[Any]: ...
    def exists(self, name: str) -> bool: ...
    def delete(self, name: str) -> None: ...
    def list(self) -> list[str]: ...

class FileSystemStorage:
    """File-based storage implementation"""
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        
    def save(self, name: str, container: ModelContainer[Any]) -> None:
        path = self.base_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(container, f)
            
    def load(self, name: str) -> ModelContainer[Any]:
        path = self.base_dir / f"{name}.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)
```

## Usage Example

```python
# Training
container = ModelContainer(
    model=booster,
    meta=ModelMeta(name="model_v1", feature_names=features)
)
storage = FileSystemStorage(Path("./models"))
storage.save("model_v1", container)

# Prediction
loaded_container = storage.load("model_v1")
predictions = loaded_container.model.predict(test_data)
```

## Design Principles
1. Pure duck typing via Protocol
2. Zero framework dependencies
3. Separate config per storage backend
4. Same interface for all model types
5. <200 lines of core code
