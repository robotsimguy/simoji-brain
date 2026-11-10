"""Loading a user `Policy` from a file, plus the adapter registry / auto-detection."""

import os
import textwrap

import numpy as np
import pytest

from simoji_policy.adapters import detect_adapter, make_adapter, resolve_adapter_name
from simoji_policy.adapters.base import MissingDependency, PolicyConfig
from simoji_policy.adapters.simoji import SimojiAdapter, load_policy_class
from simoji_policy.protocol import Hello, Obs

from conftest import JOINTS, make_hello, make_obs

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")


def _hello():
    return Hello.from_dict(make_hello())


def _obs(t=1):
    return Obs.from_dict(make_obs(t, [0.1] * 6, time=0.5))


def test_example_template_loads_and_acts():
    cfg = PolicyConfig(model=os.path.join(EXAMPLES, "simple_policy.py"), chunk=5, hz=10)
    ad = make_adapter("auto", cfg)
    assert isinstance(ad, SimojiAdapter)
    statuses = []
    ad.load(statuses.append)
    assert statuses == ["loading my model…", "ready"]
    block = ad.reset(_hello())
    assert block["name"] == "simple-sine"
    assert block["action_space"] == "joint_targets"
    assert block["joint_names"] == JOINTS
    assert block["chunk"] == 5
    out = ad.act(_obs())
    assert out.shape == (5, 6)
    assert np.all(np.abs(out - 0.1) <= 0.2 + 1e-6)


def test_minimal_policy_file(tmp_path):
    f = tmp_path / "mini.py"
    f.write_text(
        textwrap.dedent(
            """
            class Policy:
                def reset(self, hello):
                    self.n = hello.dofs
                def act(self, obs):
                    return [1.0] * self.n       # single vector -> repeated to chunk
            """
        )
    )
    ad = SimojiAdapter(PolicyConfig(model=str(f), chunk=3))
    block = ad.reset(_hello())  # reset auto-loads
    assert block["joint_names"] == JOINTS and block["name"] == "mini.py"
    out = ad.act(_obs())
    assert out.shape == (3, 6) and np.all(out == 1.0)


def test_policy_overrides_and_class_selector(tmp_path):
    f = tmp_path / "custom.py"
    f.write_text(
        textwrap.dedent(
            """
            import numpy as np
            class Walker:
                name = "walker"
                action_space = "velocity_command"
                joint_names = ["vx", "vy", "yaw"]
                def __init__(self, config):
                    self.config = config
                def reset(self, hello):
                    return {"chunk": 2, "hz": 20}
                def act(self, obs):
                    return np.ones((7, 3))      # too long -> truncated to chunk
            """
        )
    )
    ad = make_adapter("simoji", PolicyConfig(model=f"{f}:Walker", chunk=2, extra={"k": "v"}))
    block = ad.reset(_hello())
    assert block["name"] == "walker"
    assert block["action_space"] == "velocity_command"
    assert block["joint_names"] == ["vx", "vy", "yaw"]
    assert block["chunk"] == 2 and block["hz"] == 20
    assert ad.policy.config.extra == {"k": "v"}
    assert ad.act(_obs()).shape == (2, 3)


def test_bad_policy_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_policy_class(str(tmp_path / "missing.py"))
    f = tmp_path / "noclass.py"
    f.write_text("x = 1\n")
    with pytest.raises(AttributeError):
        load_policy_class(str(f))
    g = tmp_path / "noact.py"
    g.write_text("class Policy:\n    pass\n")
    with pytest.raises(TypeError):
        load_policy_class(str(g))
    with pytest.raises(ValueError):
        SimojiAdapter(PolicyConfig(model=""))


def test_detect_adapter(tmp_path):
    assert detect_adapter("") == "echo"
    assert detect_adapter("echo") == "echo"
    assert detect_adapter("zero") == "zero"
    assert detect_adapter("my_policy.py") == "simoji"
    assert detect_adapter("ws://localhost:8000") == "openpi"
    assert detect_adapter("localhost:5555") == "groot"
    assert detect_adapter("tcp://gpu:5555") == "groot"
    assert detect_adapter("openvla/openvla-7b") == "openvla"
    assert detect_adapter("nvidia/GR00T-N1.5-3B") == "groot"
    assert detect_adapter("lerobot/act_aloha_sim_transfer_cube_human") == "lerobot"
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "config.json").write_text('{"architectures": ["OpenVLAForActionPrediction"]}')
    assert detect_adapter(str(d)) == "openvla"
    assert resolve_adapter_name("auto", "x.py") == "simoji"
    with pytest.raises(ValueError):
        resolve_adapter_name("bogus", "")


@pytest.mark.parametrize("name", ["lerobot", "openvla", "groot"])
def test_heavy_adapters_give_install_hint_without_deps(name, monkeypatch):
    """Constructing is cheap; load() must raise MissingDependency with a hint if the dep is absent."""
    import builtins

    real_import = builtins.__import__
    blocked = {"lerobot", "torch", "transformers", "gr00t"}

    def fake_import(mod, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and mod.split(".")[0] in blocked:
            raise ImportError(mod)
        return real_import(mod, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    ad = make_adapter(name, PolicyConfig(model="some/model"))
    with pytest.raises(MissingDependency) as ei:
        ad.load(lambda m: None)
    assert "install" in str(ei.value).lower()


def test_openpi_adapter_needs_msgpack(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(mod, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and mod.split(".")[0] == "msgpack":
            raise ImportError(mod)
        return real_import(mod, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    ad = make_adapter("openpi", PolicyConfig(model="ws://localhost:8000"))
    with pytest.raises(MissingDependency):
        ad.load(lambda m: None)
