import pytest

from cvsm.memory import ACTIVE, RETRACTED, SUPERSEDED, Entry, SharedMemory
from cvsm.trace import Trace


def _e(mem, text, premises=()):
    return Entry(id=mem.new_id(), text=text, embedding=[], conf=0.8, label="Corroborated", support=[], evidence=[],
                 premises=list(premises), qualifiers=[], subtask="t1")


def test_append_only_and_versioning():
    tr = Trace("q")
    mem = SharedMemory(tr)
    e = mem.append(_e(mem, "A"))
    assert mem.t == 1 and e.version == 1 and e.status == ACTIVE
    with pytest.raises(ValueError):
        mem.append(e)                                      # no overwrite of an existing id
    mem.set_status(e.id, SUPERSEDED)
    assert mem.t == 2                                      # status change appends a version
    assert [v.version for v in mem.log] == [1, 2]
    assert mem.log[0].status == ACTIVE                     # old version is not mutated
    assert mem.active() == []
    writes = [ev for ev in tr.events if ev["kind"] == "memory_write"]
    assert [w["t"] for w in writes] == [1, 2] and writes[1]["active_ids"] == []


def test_transitive_retraction():
    mem = SharedMemory()
    m1 = mem.append(_e(mem, "premise"))
    m2 = mem.append(_e(mem, "derived", [m1.id]))
    m3 = mem.append(_e(mem, "derived twice", [m2.id]))
    m4 = mem.append(_e(mem, "independent"))
    mem.set_status(m1.id, SUPERSEDED)
    retracted = mem.retract_transitively(m1.id, reverify=lambda e: False)
    assert retracted == [m2.id, m3.id]
    assert mem.get(m2.id).status == RETRACTED and mem.get(m3.id).status == RETRACTED
    assert mem.get(m4.id).status == ACTIVE
    assert len(mem.log) == 4 + 1 + 2                       # every change is a new version


def test_retraction_stops_when_reverification_passes():
    mem = SharedMemory()
    m1 = mem.append(_e(mem, "premise"))
    m2 = mem.append(_e(mem, "derived", [m1.id]))
    mem.set_status(m1.id, SUPERSEDED)
    assert mem.retract_transitively(m1.id, reverify=lambda e: True) == []
    assert mem.get(m2.id).status == ACTIVE
