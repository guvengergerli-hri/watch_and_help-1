import pytest
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_TENSORIZER_PATH = REPO_ROOT / "watch" / "vae" / "tensorizer.py"
_SPEC = importlib.util.spec_from_file_location("watch_vae_tensorizer", _TENSORIZER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Unable to load tensorizer module spec.")
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

ActionCanonicalizationError = _MOD.ActionCanonicalizationError
UnknownActionKeyError = _MOD.UnknownActionKeyError
WatchGraphTensorizer = _MOD.WatchGraphTensorizer


def _make_tensorizer():
    return WatchGraphTensorizer(
        max_nodes=8,
        class_to_idx={"None": 0},
        state_to_idx={"OPEN": 0},
        action_to_idx={
            "[walk] <kitchen>": 7,
            "[open] <kitchencabinets>": 11,
        },
    )


def test_canonicalize_walktowards_with_double_spaces():
    tensorizer = _make_tensorizer()
    idx, meta = tensorizer.action_index(
        "[walktowards]  <kitchen> (11)",
        strict=True,
        return_details=True,
    )
    assert idx == 7
    assert meta["canonical_key"] == "[walk] <kitchen>"
    assert meta["normalized_action"] == "[walk] <kitchen> (11)"


def test_canonicalize_kitchencabinet_alias():
    tensorizer = _make_tensorizer()
    idx, meta = tensorizer.action_index(
        "[open]  <kitchencabinet> (140)",
        strict=True,
        return_details=True,
    )
    assert idx == 11
    assert meta["canonical_key"] == "[open] <kitchencabinets>"


def test_unknown_action_raises_structured_error():
    tensorizer = _make_tensorizer()
    with pytest.raises(UnknownActionKeyError) as exc_info:
        tensorizer.action_index(
            "[grab] <book> (12)",
            strict=True,
            source_context={"mode": "unit_test", "step_idx": 3},
        )
    payload = exc_info.value.to_dict()
    assert payload["canonical_key"] == "[grab] <book>"
    assert payload["source_context"]["mode"] == "unit_test"
    assert isinstance(payload["nearest_candidates"], list)


def test_malformed_action_raises_structured_error():
    tensorizer = _make_tensorizer()
    with pytest.raises(ActionCanonicalizationError):
        tensorizer.action_index("walk_only_token", strict=True)
