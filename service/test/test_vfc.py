""".vfc 词表文件格式测试：往返、引号选择、语法错误、契约头、原子写。

运行：cd ROOT && python -m unittest service.test.test_vfc -v
"""
from __future__ import annotations

# ---- GARGANTUA 路径引导 ----
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "start" / "boot.py").is_file():
        _sys.path.insert(0, str(_anc / "start"))
        break
else:
    raise SystemExit("找不到 GARGANTUA 项目根（start/boot.py）")
import boot
boot.bootstrap()

import shutil
import tempfile
import unittest
from pathlib import Path

import spec

try:
    from tokens import vfc
except ImportError:                                 # pragma: no cover
    # tokens/__init__ 会连带加载 vocab.py（正被并行改动）；vfc 无任何包内
    # 相对导入，退回到按文件路径导入，保证本测试不受牵连。
    import importlib.util as _ilu
    _p = _Path(__file__).resolve().parents[1] / "tokens" / "vfc.py"
    _s = _ilu.spec_from_file_location("vfc", _p)
    vfc = _ilu.module_from_spec(_s)
    _s.loader.exec_module(vfc)

NL = "\n"

# 可见区起始 ID = 架构契约头长度
_HEAD = vfc.head_entries()
_H = vfc.head_size()


def _with_extra(texts) -> list:
    """契约头 + 一批可见 token（ID 从 _H 起连续）。"""
    return list(_HEAD) + [(_H + i, t, True) for i, t in enumerate(texts)]


def _lines_of(texts) -> list:
    """只取可见区的那几行（去掉魔数行与头部 2970 行）。"""
    return vfc.dumps_vfc(_with_extra(texts)).split(NL)[1 + _H:]


class HeadContractTest(unittest.TestCase):
    """架构契约头：长度、名字、顺序、形态全部从 spec 推导。"""

    def test_head_size_matches_spec(self):
        self.assertEqual(_H, spec.HEAD_SIZE)
        self.assertEqual(_H, 6 + 3 + 2961)
        self.assertEqual(len(_HEAD), _H)

    def test_head_ids_are_dense(self):
        self.assertEqual([e[0] for e in _HEAD], list(range(_H)))

    def test_head_names_come_from_spec(self):
        # 不可显示条目的 text 自带尖括号，与 spec / vocab.json 的拼写一致
        self.assertEqual([e[1] for e in _HEAD[:6]], list(spec.SPECIAL_TOKENS))
        self.assertEqual([e[1] for e in _HEAD[6:9]], spec.SYSTEM_CHARS)
        self.assertEqual(_HEAD[9][1], "<cx:0>")
        self.assertEqual(_HEAD[_H - 1][1], "<draw:special:2>")

    def test_common_english_words_do_not_collide_with_head(self):
        """stop / model / user / caps 也是收录的英文单词，不能与头部撞文本。

        这是「不可显示条目 text 自带尖括号」的直接理由：若取裸名 'stop'，
        基础词表里那个可见英文词 'stop' 就会撞上「文本全表唯一」。
        """
        for word in ("stop", "model", "user", "caps"):
            entries = _with_extra([word])
            self.assertEqual(vfc.validate_vfc(vfc.dumps_vfc(entries)), [])
            self.assertEqual(vfc.parse_vfc(vfc.dumps_vfc(entries))[_H],
                             (_H, word, True))

    def test_head_visibility(self):
        # special / draw 不可显示；system（{ } |）不是合法标识符，只能落引号
        self.assertEqual([e[2] for e in _HEAD[:6]], [False] * 6)
        self.assertEqual([e[2] for e in _HEAD[6:9]], [True] * 3)
        self.assertEqual([e[2] for e in _HEAD[9:]], [False] * (_H - 9))

    def test_head_roundtrip_and_validates(self):
        text = vfc.dumps_vfc(_HEAD)
        self.assertEqual(vfc.parse_vfc(text), _HEAD)
        self.assertEqual(vfc.validate_vfc(text), [])

    def test_head_text_is_on_its_own_line(self):
        lines = vfc.dumps_vfc(_HEAD).split(NL)
        self.assertEqual(lines[0], vfc.MAGIC)
        self.assertEqual(lines[1], "<caps>:0")
        self.assertEqual(lines[7], '"{":6')
        self.assertEqual(lines[10], "<cx:0>:9")
        self.assertEqual(lines[1 + _H - 1], "<draw:special:2>:B99")


class RoundTripTest(unittest.TestCase):
    """dumps_vfc → parse_vfc 得到同样条目。"""

    # 覆盖：中文、emoji（非 BMP）、单引号、双引号、换行、制表符、CR、反斜杠
    _TEXTS = ["hello", "中文", "\U0001f600", "it's", 'say "hi"',
              "a" + NL + "b", "x\ty", "r\rn", "back\\slash", "<x>",
              " ", "\x00", "\x7f", " "]

    def test_roundtrip_keeps_entries(self):
        entries = _with_extra(self._TEXTS)
        self.assertEqual(vfc.parse_vfc(vfc.dumps_vfc(entries)), entries)

    def test_roundtrip_is_valid(self):
        self.assertEqual(vfc.validate_vfc(vfc.dumps_vfc(
            _with_extra(self._TEXTS))), [])

    def test_dumps_is_idempotent(self):
        entries = _with_extra(self._TEXTS)
        once = vfc.dumps_vfc(entries)
        twice = vfc.dumps_vfc(vfc.parse_vfc(once))
        self.assertEqual(once, twice)

    def test_dumps_sorts_by_id(self):
        shuffled = list(reversed(_with_extra(["a", "b"])))
        self.assertEqual(vfc.dumps_vfc(shuffled),
                         vfc.dumps_vfc(_with_extra(["a", "b"])))

    def test_output_is_one_token_per_line(self):
        entries = _with_extra(self._TEXTS)
        lines = vfc.dumps_vfc(entries).rstrip(NL).split(NL)
        # 魔数行 + 每个 token 一行（转义保证了带换行的文本不会劈成两行）
        self.assertEqual(len(lines), 1 + len(entries))

    def test_endswith_newline(self):
        self.assertTrue(vfc.dumps_vfc(_with_extra(["a"])).endswith(NL))

    def test_dumps_rejects_empty_text(self):
        with self.assertRaises(vfc.VfcError):
            vfc.dumps_vfc(_with_extra([""]))

    def test_dumps_rejects_bad_invisible_name(self):
        with self.assertRaises(vfc.VfcError):
            vfc.dumps_vfc([(0, "not a name", False)] + list(_HEAD[1:]))

    def test_dumps_rejects_id_out_of_range(self):
        with self.assertRaises(vfc.VfcError):
            vfc.dumps_vfc([(spec.MAX_VOCAB_SIZE, "x", True)])

    def test_dumps_rejects_lone_surrogate(self):
        with self.assertRaises(vfc.VfcError):
            vfc.dumps_vfc(_with_extra(["\ud800"]))

    def test_invisible_and_visible_same_text_conflict(self):
        """<stop>:5 与 "<stop>":N 文本相同，唯一性规则下不能共存。

        两种形态是**不同的 token**（区别由 visible 标记保留，各自单独往返
        无损），但反查映射里同文本两个 ID 是歧义，所以拒绝同文件共存。
        """
        errs = vfc.validate_vfc(vfc.dumps_vfc(_with_extra(["<stop>"])))
        self.assertTrue(any("文本重复" in e["message"] for e in errs), errs)

    def test_visible_text_that_merely_looks_invisible_is_fine_alone(self):
        """只有可显示形态的 "<stop>" 时（不与头部同文本冲突的写法）能正常存。"""
        entries = _with_extra(["<stop-x>"])
        self.assertEqual(vfc.parse_vfc(vfc.dumps_vfc(entries))[_H],
                         (_H, "<stop-x>", True))


class QuoteChoiceTest(unittest.TestCase):
    """序列化时的引号选择。"""

    def test_default_is_double_quote(self):
        self.assertEqual(_lines_of(["hello"])[0], '"hello":%X' % _H)

    def test_double_quote_in_text_uses_single(self):
        self.assertEqual(_lines_of(['wor"ld'])[0], "'wor\"ld':%X" % _H)

    def test_single_quote_in_text_uses_double(self):
        self.assertEqual(_lines_of(["it's"])[0], '"it\'s":%X' % _H)

    def test_both_quotes_uses_double_and_escapes(self):
        self.assertEqual(_lines_of(['he"llo\'w'])[0],
                         '"he\\"llo\'w":%X' % _H)

    def test_unclosed_quote_is_error(self):
        errs = vfc.validate_vfc(vfc.MAGIC + NL + '"unterminated:%X' % _H)
        self.assertTrue(any("引号未闭合" in e["message"] for e in errs))

    def test_control_chars_are_escaped(self):
        lines = _lines_of(["a" + NL + "b", "x\ty", "r\rn", "\x00"])
        self.assertEqual(lines[0], '"a\\nb":%X' % _H)
        self.assertEqual(lines[1], '"x\\ty":%X' % (_H + 1))
        self.assertEqual(lines[2], '"r\\rn":%X' % (_H + 2))
        self.assertEqual(lines[3], '"\\u0000":%X' % (_H + 3))

    def test_backslash_and_u_escape_roundtrip(self):
        entries = _with_extra(["back\\slash", "é", "\U0001f600"])
        back = vfc.parse_vfc(vfc.dumps_vfc(entries))
        self.assertEqual(back[_H][1], "back\\slash")
        self.assertEqual(back[_H + 1][1], "é")
        self.assertEqual(back[_H + 2][1], "\U0001f600")

    def test_surrogate_pair_escape(self):
        text = vfc.MAGIC + NL + NL.join(
            vfc.dumps_vfc(_HEAD).split(NL)[1:]) + NL + '"\\ud83d\\ude00":%X' % _H
        self.assertEqual(vfc.parse_vfc(text)[_H][1], "\U0001f600")


class SyntaxErrorTest(unittest.TestCase):
    """每类语法错误都要被拒绝，且给出行号。"""

    def _one_bad_line(self, bad_line: str):
        """魔数行 + 一个坏行 → 错误列表。"""
        return vfc.validate_vfc(vfc.MAGIC + NL + bad_line + NL)

    def _assert_rejected(self, bad_line, keyword):
        errs = self._one_bad_line(bad_line)
        self.assertTrue(errs, f"{bad_line!r} 应当被拒绝")
        self.assertTrue(
            any(keyword in e["message"] for e in errs),
            f"{bad_line!r} 的错误里应含 {keyword!r}，实际：{errs}")
        # 行号必须落在坏行上（魔数行是第 1 行，坏行是第 2 行）
        self.assertTrue(any(e["line"] == 2 for e in errs), errs)

    def test_missing_colon(self):
        self._assert_rejected('"hello"', "缺少")

    def test_id_not_a_number(self):
        # xyz 含非 hex 字符（x）→ 拒绝；abc 全是 hex 字符，现在是合法 ID
        self._assert_rejected('"hello":xyz', "数字")

    def test_space_after_colon(self):
        self._assert_rejected('"hello": 12', "数字")

    def test_negative_id(self):
        self._assert_rejected('"hello":-1', "数字")

    def test_unclosed_quote(self):
        self._assert_rejected('"hello:%X' % _H, "引号未闭合")

    def test_mismatched_quotes(self):
        self._assert_rejected('"hello\':%X' % _H, "引号未闭合")

    def test_unclosed_angle_bracket(self):
        self._assert_rejected("<model:%d" % _H, "尖括号未闭合")

    def test_illegal_escape(self):
        self._assert_rejected('"a\\db":%X' % _H, "非法转义")

    def test_illegal_escape_b_and_f(self):
        # \b \f \/ 是 JSON 接受的，但 .vfc 严格模式一律拒绝
        self._assert_rejected('"a\\bb":%X' % _H, "非法转义")
        self._assert_rejected('"a\\fb":%X' % _H, "非法转义")

    def test_trailing_backslash(self):
        self._assert_rejected('"ab\\":%X' % _H, "反斜杠")

    def test_empty_text(self):
        self._assert_rejected('"":%X' % _H, "不得为空")

    def test_illegal_angled_name(self):
        self._assert_rejected("<1bad>:%d" % _H, "名字非法")
        self._assert_rejected("<a b>:%d" % _H, "名字非法")
        self._assert_rejected("<a>b>:%d" % _H, "名字非法")

    def test_bare_text_without_wrapper(self):
        self._assert_rejected("hello:%d" % _H, "包裹")

    def test_id_over_upper_bound(self):
        self._assert_rejected('"x":%X' % spec.MAX_VOCAB_SIZE, "越界")

    def test_lone_low_surrogate(self):
        self._assert_rejected('"\\udc00":%X' % _H, "代理")

    def test_bad_u_escape(self):
        self._assert_rejected('"\\uZZZZ":%X' % _H, "十六进制")
        self._assert_rejected('"\\u12":%X' % _H, "十六进制")

    def test_parse_vfc_raises_with_line(self):
        with self.assertRaises(vfc.VfcError) as cm:
            vfc.parse_vfc(vfc.MAGIC + NL + '"hello":xyz' + NL)
        self.assertEqual(cm.exception.line, 2)
        self.assertIsInstance(cm.exception, ValueError)

    def test_parse_vfc_error_has_source(self):
        with self.assertRaises(vfc.VfcError) as cm:
            vfc.parse_vfc(vfc.MAGIC + NL + "oops" + NL, source="my.vfc")
        self.assertIn("my.vfc", str(cm.exception))


class SemanticErrorTest(unittest.TestCase):
    """重复 ID / 重复文本 / ID 不连续。"""

    def test_duplicate_id(self):
        entries = _with_extra(["a", "b"])
        entries[-1] = (entries[-2][0], "b", True)     # 两个条目同 ID
        errs = vfc.validate_vfc(vfc.dumps_vfc(entries))
        self.assertTrue(any("ID 重复" in e["message"] for e in errs), errs)

    def test_duplicate_text(self):
        errs = vfc.validate_vfc(vfc.dumps_vfc(_with_extra(["dup", "dup"])))
        self.assertTrue(any("文本重复" in e["message"] for e in errs), errs)

    def test_id_hole_is_reported(self):
        # 合法文件删掉中间一行 → ID 出现空位
        lines = vfc.dumps_vfc(_with_extra(["a", "b", "c"])).rstrip(NL).split(NL)
        del lines[1 + _H + 1]                        # 删掉 ID=_H+1 那一行
        errs = vfc.validate_vfc(NL.join(lines) + NL)
        self.assertTrue(any("缺少 ID" in e["message"] for e in errs), errs)

    def test_id_gap_at_the_very_end(self):
        entries = _with_extra(["a"]) + [(_H + 5, "b", True)]
        errs = vfc.validate_vfc(vfc.dumps_vfc(entries))
        self.assertTrue(any("缺少 ID" in e["message"] for e in errs), errs)

    def test_valid_file_has_no_errors(self):
        self.assertEqual(vfc.validate_vfc(
            vfc.dumps_vfc(_with_extra(["a", "b", "c"]))), [])


class HeadViolationTest(unittest.TestCase):
    """头部契约不符：顺序错、缺头部、头部 ID 不对。"""

    def _head_lines(self):
        return vfc.dumps_vfc(_HEAD).rstrip(NL).split(NL)[1:]

    def test_head_order_swapped(self):
        lines = self._head_lines()
        # 只换名字、不换 ID（换整行的话排序后条目不变，测不出顺序错）
        lines[0], lines[1] = "<user>:0", "<caps>:1"
        errs = vfc.validate_vfc(vfc.MAGIC + NL + NL.join(lines) + NL)
        self.assertTrue(any("架构契约头" in e["message"] for e in errs), errs)

    def test_head_text_wrong(self):
        lines = self._head_lines()
        lines[0] = '"<caps>":0'                      # 误写成可显示形态
        errs = vfc.validate_vfc(vfc.MAGIC + NL + NL.join(lines) + NL)
        self.assertTrue(any("架构契约头" in e["message"] for e in errs), errs)

    def test_missing_head(self):
        errs = vfc.validate_vfc(vfc.MAGIC + NL + '"hello":0' + NL)
        self.assertTrue(any("架构契约头不完整" in e["message"] for e in errs),
                        errs)

    def test_head_id_shifted(self):
        lines = self._head_lines()
        shifted = []
        for line in lines:
            name, _, tid = line.rpartition(":")
            # ID 是十六进制（2026-10-31）：按 16 进制解析、16 进制写回
            shifted.append(f"{name}:{int(tid, 16) + 1:X}")
        errs = vfc.validate_vfc(vfc.MAGIC + NL + NL.join(shifted) + NL)
        self.assertTrue(any("架构契约头第 1 条 ID 应为 0" in e["message"]
                            for e in errs), errs)

    def test_head_errors_are_truncated(self):
        # 头部整体错位时不能报出 2970 条
        lines = self._head_lines()[1:] + ["<caps>:0"]
        errs = vfc.validate_vfc(vfc.MAGIC + NL + NL.join(lines) + NL)
        head_errs = [e for e in errs if "架构契约头" in e["message"]]
        self.assertLessEqual(len(head_errs), vfc._MAX_HEAD_ERRORS + 1)


class LenientReadTest(unittest.TestCase):
    """注释行与魔数行的宽容处理。"""

    def test_magic_line_is_optional(self):
        body = NL.join(self._head_lines()) + NL
        self.assertEqual(vfc.validate_vfc(body), [])
        self.assertEqual(vfc.parse_vfc(body), _HEAD)

    def test_comment_lines_are_ignored(self):
        text = (vfc.MAGIC + NL + "# 这是一条注释" + NL
                + NL.join(self._head_lines()) + NL + "# 尾部注释" + NL)
        self.assertEqual(vfc.validate_vfc(text), [])
        self.assertEqual(vfc.parse_vfc(text), _HEAD)

    def test_blank_lines_are_ignored(self):
        text = vfc.MAGIC + NL + NL + NL.join(self._head_lines()) + NL + NL
        self.assertEqual(vfc.validate_vfc(text), [])
        self.assertEqual(vfc.parse_vfc(text), _HEAD)

    def test_crlf_is_tolerated(self):
        text = vfc.dumps_vfc(_HEAD).replace(NL, "\r\n")
        self.assertEqual(vfc.validate_vfc(text), [])
        self.assertEqual(vfc.parse_vfc(text), _HEAD)

    def test_wrong_magic_is_treated_as_comment(self):
        # 魔数不对也只当注释跳过（宽容读取），由调用方决定是否检查
        text = "#some-other-magic" + NL + NL.join(self._head_lines()) + NL
        self.assertEqual(vfc.validate_vfc(text), [])

    def test_empty_file_reports_missing_head(self):
        errs = vfc.validate_vfc("")
        self.assertTrue(any("架构契约头不完整" in e["message"] for e in errs))

    def _head_lines(self):
        return vfc.dumps_vfc(_HEAD).rstrip(NL).split(NL)[1:]


class ValidateMultiErrorTest(unittest.TestCase):
    """validate_vfc 一次报多条错误。"""

    def test_reports_every_bad_line(self):
        text = (vfc.MAGIC + NL
                + '"a":0' + NL          # 第 2 行：语法没问题但头部不对
                + "no-colon" + NL       # 第 3 行：缺冒号
                + '"b":xyz' + NL)       # 第 4 行：ID 非数字
        errs = vfc.validate_vfc(text)
        lines = {e["line"] for e in errs}
        self.assertIn(3, lines)
        self.assertIn(4, lines)

    def test_reports_duplicate_id_and_text_and_hole_together(self):
        entries = _with_extra(["dup"]) + [
            (_H + 1, "dup", True),                  # 文本重复
            (_H + 1, "other", True),                # ID 重复
            (_H + 5, "far", True),                  # 造成空位
        ]
        errs = vfc.validate_vfc(vfc.dumps_vfc(entries))
        msgs = " | ".join(e["message"] for e in errs)
        self.assertIn("ID 重复", msgs)
        self.assertIn("文本重复", msgs)
        self.assertTrue(any("缺少 ID" in m for m in
                            (e["message"] for e in errs)), msgs)
        self.assertGreaterEqual(len(errs), 3)

    def test_error_entries_have_line_and_message(self):
        errs = vfc.validate_vfc(vfc.MAGIC + NL + "oops" + NL)
        for e in errs:
            self.assertIsInstance(e["message"], str)
            self.assertTrue(e["line"] is None or isinstance(e["line"], int))

    def test_parse_vfc_raises_on_first_error_only(self):
        with self.assertRaises(vfc.VfcError):
            vfc.parse_vfc(vfc.MAGIC + NL + "oops" + NL + "oops2" + NL)


class FileIoTest(unittest.TestCase):
    """read_vfc / write_vfc：原子写后文件可读，且不留临时文件。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="garg_vfc_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_write_then_read(self):
        p = self.tmp / "vocab.vfc"
        entries = _with_extra(["hello", "中文", "\U0001f600", "a" + NL + "b"])
        vfc.write_vfc(str(p), entries)
        self.assertEqual(vfc.read_vfc(str(p)), entries)

    def test_write_is_atomic_and_leaves_no_tmp(self):
        p = self.tmp / "vocab.vfc"
        entries = _with_extra(["a", "b"])
        vfc.write_vfc(str(p), entries)
        leftovers = [f.name for f in self.tmp.iterdir() if ".tmp" in f.name]
        self.assertEqual(leftovers, [])

    def test_write_overwrites_existing_file(self):
        p = self.tmp / "vocab.vfc"
        vfc.write_vfc(str(p), _with_extra(["a", "b", "c"]))
        entries = _with_extra(["x"])
        vfc.write_vfc(str(p), entries)
        self.assertEqual(vfc.read_vfc(str(p)), entries)

    def test_written_file_starts_with_magic(self):
        p = self.tmp / "vocab.vfc"
        vfc.write_vfc(str(p), _with_extra(["a"]))
        self.assertEqual(p.read_text(encoding="utf-8").split(NL)[0],
                         vfc.MAGIC)

    def test_written_file_is_utf8(self):
        p = self.tmp / "vocab.vfc"
        vfc.write_vfc(str(p), _with_extra(["中文"]))
        raw = p.read_bytes()
        self.assertIn("中文".encode("utf-8"), raw)

    def test_read_missing_file_raises(self):
        with self.assertRaises(OSError):
            vfc.read_vfc(str(self.tmp / "nope.vfc"))

    def test_read_rejects_invalid_file(self):
        p = self.tmp / "bad.vfc"
        p.write_text(vfc.MAGIC + NL + "oops" + NL, encoding="utf-8")
        with self.assertRaises(vfc.VfcError):
            vfc.read_vfc(str(p))


if __name__ == "__main__":
    unittest.main()
