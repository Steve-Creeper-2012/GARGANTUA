# ===========================================================================
# GARGANTUA v1 — MIDI codec（纯 Python SMF 读写 ↔ MIDI token 序列）
# ---------------------------------------------------------------------------
# 契约来源：ROOT/spec.py（冻结，禁止修改）MIDI 区间注释：
#   绝对 ID = spec.MIDI_BEGIN + 偏移：
#     0–127   note_on(pitch)      128–255 note_off(pitch)
#     256–287 velocity 32 档      288–415 program(0–127)
#     416–447 tempo 32 档 40–240BPM
#     448–547 time_shift 100 档，10ms 步进，最长 1s，超过拆多个
#   事件顺序：time_shift 表达事件间隔；note_on 后必须紧跟一个 velocity
#   档 token（合并 note_on+velocity 的解码语义）。
# 纯标准库 SMF 解析：header/track chunk、running status、变长 delta time、
# note on/off、program change、set tempo；忽略其他 meta/event，不报错。
# tokens_to_midi 写出合法 SMF：format 0、单轨、480 ticks/quarter。
# ===========================================================================
"""midi_codec: Standard MIDI File ↔ MIDI token 序列。"""

from __future__ import annotations

import os
import struct
from typing import List, Sequence, Tuple, Union

import spec

TPQ = 480                                       # ticks per quarter（写出）
DEFAULT_BPM = 120.0

# 相对偏移（见 spec.py MIDI 区间注释）
OFF_NOTE_ON, OFF_NOTE_OFF = 0, 128
OFF_VEL, OFF_PROG, OFF_TEMPO, OFF_TIME = 256, 288, 416, 448
VEL_LEVELS = 32
TEMPO_LEVELS, TEMPO_MIN, TEMPO_MAX = 32, 40.0, 240.0
TIME_LEVELS, TIME_STEP_MS, TIME_MAX_MS = 100, 10, 1000

# 事件 = (time_ms, kind, a, b)
# 排序约定：全程使用仅按时间的【稳定排序】——token 流顺序即事件顺序，
# 写出文件保持该顺序，读回稳定排序不打乱同毫秒事件，保证 token 级
# roundtrip 逐一致。
#   on:   a=pitch, b=velocity(1..127)
#   off:  a=pitch, b=unused
#   prog: a=program
#   tempo: a=us_per_quarter
Event = Tuple[float, str, int, int]


# ---------------------------------------------------------------------------
# 量化映射（独立函数，便于测试）
# ---------------------------------------------------------------------------
def vel_to_q(vel: int) -> int:
    """velocity 0..127 → 32 档。"""
    return min(VEL_LEVELS - 1, max(0, vel) >> 2)


def q_to_vel(q: int) -> int:
    """32 档 → velocity 代表值（量化不动点，保证 roundtrip 一致）。"""
    return min(127, q * 4 + 2)


def bpm_to_q(bpm: float) -> int:
    """40–240 BPM → 32 档。"""
    q = int(round((bpm - TEMPO_MIN) * (TEMPO_LEVELS - 1)
                  / (TEMPO_MAX - TEMPO_MIN)))
    return max(0, min(TEMPO_LEVELS - 1, q))


def q_to_bpm(q: int) -> float:
    """32 档 → BPM 代表值。"""
    q = max(0, min(TEMPO_LEVELS - 1, q))
    return TEMPO_MIN + q * (TEMPO_MAX - TEMPO_MIN) / (TEMPO_LEVELS - 1)


def ms_to_time_tokens(delta_ms: float) -> List[int]:
    """时间间隔 → time_shift token 列表（>1s 拆多个）。"""
    out: List[int] = []
    total = int(round(delta_ms))
    while total >= TIME_MAX_MS:
        out.append(spec.MIDI_BEGIN + OFF_TIME + TIME_LEVELS - 1)
        total -= TIME_MAX_MS
    k = int(total / TIME_STEP_MS + 0.5) - 1
    if k >= 0:
        out.append(spec.MIDI_BEGIN + OFF_TIME + min(k, TIME_LEVELS - 1))
    return out


def time_token_to_ms(rel: int) -> int:
    """time_shift 相对偏移 → 毫秒。"""
    return (rel + 1) * TIME_STEP_MS


# ---------------------------------------------------------------------------
# SMF 解析
# ---------------------------------------------------------------------------
def _read_varlen(data: bytes, i: int) -> Tuple[int, int]:
    val = 0
    while True:
        b = data[i]
        i += 1
        val = (val << 7) | (b & 0x7F)
        if not (b & 0x80):
            return val, i


def _parse_track(data: bytes) -> List[Tuple[int, str, int, int]]:
    """单轨字节流 → [(abs_tick, kind, a, b)]；未知事件跳过不报错。"""
    events: List[Tuple[int, str, int, int]] = []
    i, tick, status = 0, 0, 0
    n = len(data)
    while i < n:
        delta, i = _read_varlen(data, i)
        tick += delta
        b = data[i]
        if b >= 0x80:
            status = b
            i += 1
        # else: running status，复用上一个 status
        if status == 0xFF:                              # meta event
            mtype = data[i]
            i += 1
            length, i = _read_varlen(data, i)
            if mtype == 0x51 and length == 3:           # set tempo
                us = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2]
                if us > 0:
                    events.append((tick, "tempo", us, 0))
            i += length
            if mtype == 0x2F:                           # end of track
                break
        elif status in (0xF0, 0xF7):                    # sysex
            length, i = _read_varlen(data, i)
            i += length
        else:
            hi = status & 0xF0
            if hi == 0x80:                              # note off
                pitch, _vel = data[i], data[i + 1]
                i += 2
                events.append((tick, "off", pitch, 0))
            elif hi == 0x90:                            # note on
                pitch, vel = data[i], data[i + 1]
                i += 2
                if vel == 0:                            # 惯例：vel=0 即 off
                    events.append((tick, "off", pitch, 0))
                else:
                    events.append((tick, "on", pitch, vel))
            elif hi == 0xC0:                            # program change
                events.append((tick, "prog", data[i], 0))
                i += 1
            elif hi == 0xD0:                            # channel pressure
                i += 1
            else:                                       # 0xA0/0xB0/0xE0 等
                i += 2
    return events


def parse_midi(path: Union[str, os.PathLike]) -> List[Event]:
    """解析 SMF → 毫秒时间轴事件列表（按时间、类型排序）。"""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"MThd":
        raise ValueError("not a Standard MIDI File")
    hlen = struct.unpack(">I", data[4:8])[0]
    _fmt, _ntrks, division = struct.unpack(">HHH", data[8:14])
    if division & 0x8000:
        raise ValueError("SMPTE time division not supported")
    tpq = division
    # 收集所有轨事件（tick 轴）
    tick_events: List[Tuple[int, str, int, int]] = []
    i = 8 + hlen
    while i + 8 <= len(data):
        tag, clen = data[i:i + 4], struct.unpack(">I", data[i + 4:i + 8])[0]
        body = data[i + 8:i + 8 + clen]
        if tag == b"MTrk":
            tick_events.extend(_parse_track(body))
        i += 8 + clen
    # tick 轴 → 毫秒轴（遍历维护 tempo；order 保证同 tick 稳定序）
    tick_events.sort(key=lambda e: e[0])
    out: List[Event] = []
    bpm = DEFAULT_BPM
    last_tick = 0
    ms = 0.0
    for tick, kind, a, b in tick_events:
        ms += (tick - last_tick) * 60000.0 / (bpm * tpq)
        last_tick = tick
        out.append((ms, kind, a, b))
        if kind == "tempo":
            bpm = 60000000.0 / a
    out.sort(key=lambda e: e[0])                # 稳定：保持文件内同刻顺序
    return out


# ---------------------------------------------------------------------------
# 事件 ↔ token
# ---------------------------------------------------------------------------
def events_to_tokens(events: Sequence[Event]) -> List[int]:
    """毫秒轴事件 → MIDI token（事件间插 time_shift）。"""
    ids: List[int] = []
    last_ms = 0.0
    for ms, kind, a, b in events:
        delta = ms - last_ms
        if delta > 0:
            ids.extend(ms_to_time_tokens(delta))
        last_ms = ms
        if kind == "on":
            ids.append(spec.MIDI_BEGIN + OFF_NOTE_ON + a)
            ids.append(spec.MIDI_BEGIN + OFF_VEL + vel_to_q(b))
        elif kind == "off":
            ids.append(spec.MIDI_BEGIN + OFF_NOTE_OFF + a)
        elif kind == "prog":
            ids.append(spec.MIDI_BEGIN + OFF_PROG + a)
        elif kind == "tempo":
            ids.append(spec.MIDI_BEGIN + OFF_TEMPO
                       + bpm_to_q(60000000.0 / a))
    return ids


def tokens_to_events(ids: Sequence[int]) -> List[Event]:
    """MIDI token → 毫秒轴事件。note_on 合并其后的 velocity 档 token。

    MIDI 区间以外的 token 忽略；无 velocity 跟随的 note_on 以默认
    velocity 档（96）补全，保证不丢音符。
    """
    events: List[Event] = []
    ms = 0.0
    pending_pitch: int | None = None

    def flush_pending() -> None:
        nonlocal pending_pitch
        if pending_pitch is not None:
            events.append((ms, "on", pending_pitch, q_to_vel(24)))
            pending_pitch = None

    for tid in ids:
        if not (spec.MIDI_BEGIN <= tid < spec.MIDI_BEGIN + spec.MIDI_SIZE):
            continue
        rel = tid - spec.MIDI_BEGIN
        if rel < OFF_NOTE_OFF:                          # note_on
            flush_pending()
            pending_pitch = rel
        elif rel < OFF_VEL:                             # note_off
            flush_pending()
            events.append((ms, "off", rel - OFF_NOTE_OFF, 0))
        elif rel < OFF_PROG:                            # velocity
            if pending_pitch is not None:
                events.append((ms, "on", pending_pitch,
                               q_to_vel(rel - OFF_VEL)))
                pending_pitch = None
        elif rel < OFF_TEMPO:                           # program
            flush_pending()
            events.append((ms, "prog", rel - OFF_PROG, 0))
        elif rel < OFF_TIME:                            # tempo
            flush_pending()
            bpm = q_to_bpm(rel - OFF_TEMPO)
            events.append((ms, "tempo", int(round(60000000.0 / bpm)), 0))
        elif rel < OFF_TIME + TIME_LEVELS:              # time_shift
            flush_pending()
            ms += time_token_to_ms(rel - OFF_TIME)
    flush_pending()
    # 不重排：token 流即时间顺序（time_shift 只增不减）
    return events


def midi_to_tokens(path: Union[str, os.PathLike]) -> List[int]:
    """SMF 文件 → MIDI token 序列。"""
    return events_to_tokens(parse_midi(path))


# ---------------------------------------------------------------------------
# SMF 写出（format 0，单轨，480 ticks/quarter）
# ---------------------------------------------------------------------------
def _write_varlen(val: int) -> bytes:
    val = max(0, int(val))
    out = bytearray([val & 0x7F])
    val >>= 7
    while val:
        out.insert(0, (val & 0x7F) | 0x80)
        val >>= 7
    return bytes(out)


def tokens_to_midi(ids: Sequence[int],
                   path: Union[str, os.PathLike]) -> List[Event]:
    """MIDI token → 合法 SMF 文件（format 0、单轨、480 ticks/quarter）。

    返回写出的事件列表（供调试/测试）。
    """
    events = tokens_to_events(ids)
    track = bytearray()
    bpm = DEFAULT_BPM
    last_ms = 0.0
    for ms, kind, a, b in events:
        delta_ticks = int(round((ms - last_ms) * bpm * TPQ / 60000.0))
        track += _write_varlen(delta_ticks)
        last_ms = ms
        if kind == "on":
            track += bytes((0x90, a & 0x7F, max(1, b) & 0x7F))
        elif kind == "off":
            track += bytes((0x80, a & 0x7F, 64))
        elif kind == "prog":
            track += bytes((0xC0, a & 0x7F))
        elif kind == "tempo":
            track += b"\xFF\x51\x03" + a.to_bytes(3, "big")
            bpm = 60000000.0 / a
    track += b"\x00\xFF\x2F\x00"                        # end of track
    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, TPQ)
    chunk = b"MTrk" + struct.pack(">I", len(track)) + bytes(track)
    with open(path, "wb") as f:
        f.write(header + chunk)
    return events
