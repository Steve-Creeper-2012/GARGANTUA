# ===========================================================================
# GARGANTUA v1 — 图元编辑器（前端）与 draw_codec（后端）的编码对拍
# ---------------------------------------------------------------------------
# 前端「图元编辑器」要在浏览器里**实时**算 token ID（改一个字段立刻出
# token），这必然把「值 → bin」的算术在 JS 里再实现一遍。两边一旦算错
# 一位，用户照着编辑器抄下来的 token 就是废的——而且很难看出来（图还是
# 照画，只是位置差半个像素）。
#
# 所以这里做**对拍**：从 service/web/app.js 里原样抠出 pdrawTokenText /
# _pdBin 两个函数（不是抄一份），用同一批图元分别喂给 JS 与 Python，
# 要求 ID 序列逐位相等。
#
# 覆盖的边界：画布 token 打头、theta ≥ 90° 交换宽高、cx/cy 到 1919/1079、
# w/h 到画布上界、颜色 10bit（0..1023）、背景图元（几何全 0）。
#
# 前置：node 可用（前端构建门 minify.py --check 已经要求它）；没有就跳过。
# ===========================================================================
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# ---- 路径引导（server/ 与 service/* 上 path）----
for _anc in Path(__file__).resolve().parents:
    if (_anc / "start" / "boot.py").is_file():
        sys.path.insert(0, str(_anc / "start"))
        break
else:
    raise SystemExit("找不到 GARGANTUA 项目根（start/boot.py）")
import boot  # noqa: E402

boot.bootstrap()

import spec  # noqa: E402
from tokens import draw_codec  # noqa: E402
from server import draw_layout_payload  # noqa: E402


def _root() -> Path:
    for anc in Path(__file__).resolve().parents:
        if (anc / "start" / "boot.py").is_file():
            return anc
    raise RuntimeError("找不到项目根")


ROOT = _root()
APP_JS = ROOT / "service" / "web" / "app.js"
# 2026-11-04：对拍也只用项目内下载的 node，绝不依赖系统 PATH
from paths import bundled_tool
NODE = bundled_tool("node") or None

# 前端从 /api/status.draw_layout 拿到的就是这张表——直接 import server.py
# 的唯一构造函数，保证测试与线上永远是同一份（改 spec 不用改这里）。
LAYOUT = draw_layout_payload()

CASES = [
    # 普通矩形 + 20°
    [dict(cx=128, cy=96, w=40, h=20, shape_type=0,
          theta=math.radians(20), r=254, g=0, b=128)],
    # 椭圆 + 100°（≥90° 要交换宽高）+ 大画布边界 + 非整数坐标
    [dict(cx=1500.5, cy=900.25, w=127.75, h=63.5, shape_type=1,
          theta=math.radians(100), r=13, g=200, b=255)],
    # 背景（几何全 0）+ 右下角极限 + 179°
    [dict(cx=0, cy=0, w=0, h=0, shape_type=-1, theta=0.0,
          r=10, g=20, b=30),
     dict(cx=1919, cy=1079, w=1, h=1, shape_type=0,
          theta=math.radians(179), r=255, g=255, b=255)],
]

_JS_HARNESS = r"""
const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8');
const data = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));

// 从真实源码里抠出函数体（保证测的是页面上跑的那份代码，不是抄本）
function grab(name) {
  const i = src.indexOf('function ' + name + '(');
  if (i < 0) throw new Error('app.js 里找不到 ' + name);
  let depth = 0, j = src.indexOf('{', i), start = j;
  for (; j < src.length; j++) {
    if (src[j] === '{') depth++;
    else if (src[j] === '}') { depth--; if (depth === 0) { j++; break; } }
  }
  return src.slice(i, j);
}
const _IDHEX = n => Number(n).toString(16).toUpperCase();
const PDRAW = { w: 1920, h: 1080, prims: [], sel: -1,
                layout: data.layout, canvasList: data.canvasList, brush: {} };
eval(grab('_pdBin'));
eval(grab('pdrawTokenText'));

const out = [];
for (const c of data.cases) {
  PDRAW.prims = c;
  const txt = pdrawTokenText();
  const ids = [];
  txt.split('}').filter(Boolean).forEach(seg => {
    seg.slice(1).split('|').forEach(p => {
      const m = /text:\s*([0-9A-F]+)/.exec(p.trim());
      if (m) ids.push(parseInt(m[1], 16));
    });
  });
  out.push(ids);
}
process.stdout.write(JSON.stringify(out));
"""


@unittest.skipIf(NODE is None, "node 不可用，跳过前端/后端编码对拍")
class TestPdrawJsParity(unittest.TestCase):
    """图元编辑器的 JS 编码必须与 draw_codec 逐位一致。"""

    def test_js_tokens_match_python(self):
        expect = [
            draw_codec.primitives_to_tokens(
                [draw_codec.Primitive(**c) for c in case], canvas=(1920, 1080))
            for case in CASES
        ]
        with tempfile.TemporaryDirectory() as td:
            js = Path(td) / "harness.js"
            js.write_text(_JS_HARNESS, encoding="utf-8")
            data = Path(td) / "case.json"
            data.write_text(json.dumps({
                "layout": LAYOUT,
                "canvasList": [list(s) for s in spec.DRAW_CANVAS_SIZES],
                "cases": CASES,
            }), encoding="utf-8")
            r = subprocess.run([NODE, str(js), str(APP_JS), str(data)],
                               capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0,
                         f"node 执行失败: {r.stderr[-800:]}")
        got = json.loads(r.stdout)
        self.assertEqual(len(got), len(expect))
        for i, (g, e) in enumerate(zip(got, expect)):
            self.assertEqual(g, e, f"第 {i} 组用例 JS 与 Python 编码不一致")

    def test_layout_matches_spec(self):
        """/api/status.draw_layout 的字段表必须覆盖 9 个图元字段。"""
        names = [f["name"] for f in LAYOUT["fields"]]
        self.assertEqual(tuple(names), spec.DRAW_FIELD_ORDER)
        for f in LAYOUT["fields"]:
            self.assertLess(f["offset"], spec.DRAW_VOCAB_SIZE)
        self.assertEqual(LAYOUT["canvas"]["offset"], spec.DRAW_CANVAS_OFFSET)
        self.assertEqual(LAYOUT["begin"], spec.DRAW_BEGIN)


if __name__ == "__main__":
    unittest.main()
