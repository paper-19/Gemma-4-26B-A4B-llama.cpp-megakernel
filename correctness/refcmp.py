"""Compare a candidate llama-refdump run against a reference run.

usage: refcmp.py <base dir> <cand dir> [--json out.json] [--policy policy.json [--policy-set NAME]]
                 [--per-position] [--quiet]

Both dirs hold one subdir per prompt (tokens.i32, logits.f32, meta.json, optional t.<name>.bin and kv.bin);
the candidate must have been run with --tokens-from <base dir>, so every position sees the same input.

Per position (the prediction made after decoding that token):
  kld        KL(P_base || P_cand) in nats, softmax over the full vocabulary in float64
  top1       argmax agreement; "confident flips" are disagreements where the base's top-1 probability > 0.5
  logit_rms  RMS of the logit difference over the vocabulary (logit_rms_centered: after removing its mean, which
             the softmax ignores; reported, not used by policy.json)
Positions are reported for all positions and for the continuation (positions >= n_prompt - 1, i.e. the
predictions of generated tokens, which is what chat use sees). For teacher-forced candidates, "greedy" counts
the prompts whose greedy continuation would stay identical and where the others would first diverge.
Watched tensors: relative L2 error ||cand - base|| / ||base|| per position, aggregated per tensor family
(name without the -<layer> suffix) and per layer; ffn_moe_topk (int32 expert ids) gives the routing flip
rate (fraction of layer x position slots whose selected expert set differs), and the logit metrics are split
by routing: positions where no layer's expert set differs at that position or any earlier one ("clean")
against the rest. kv.bin is compared as float16 words (the state format is mostly the f16 K/V cache).
--per-position adds per-position arrays (KLD, top-1 agreement, base top-1 probability, layers whose routing
differs) to the JSON for each prompt.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROW_CHUNK = 32


def read_meta(d):
    # the continuation text is cut at 64 tokens and may end inside a multi-byte character
    return json.loads((d / "meta.json").read_bytes().decode("utf-8", "replace"))


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def compare_logits(bdir, cdir, mb, mc):
    """Over the positions both runs logged (a --prefill run logs from its last prompt token on)."""
    V = mb["n_vocab"]
    assert mc["n_vocab"] == V, (bdir, V, mc["n_vocab"])
    start = max(mb["logits_from"], mc["logits_from"])
    end = min(mb["logits_from"] + mb["n_logits"], mc["logits_from"] + mc["n_logits"])
    n = max(end - start, 0)
    lb = np.memmap(bdir / "logits.f32", dtype=np.float32, mode="r", shape=(mb["n_logits"], V))[start - mb["logits_from"]:][:n]
    lc = np.memmap(cdir / "logits.f32", dtype=np.float32, mode="r", shape=(mc["n_logits"], V))[start - mc["logits_from"]:][:n]
    kld, agree, conf_flip, rms, rmsc, p1 = (np.empty(n) for _ in range(6))
    for i in range(0, n, ROW_CHUNK):
        a, b = log_softmax(lb[i:i + ROW_CHUNK]), log_softmax(lc[i:i + ROW_CHUNK])
        pa = np.exp(a)
        kld[i:i + ROW_CHUNK] = np.maximum((pa * (a - b)).sum(axis=-1), 0.0)
        ta, tb = a.argmax(axis=-1), b.argmax(axis=-1)
        agree[i:i + ROW_CHUNK] = ta == tb
        top = pa.max(axis=-1)
        p1[i:i + ROW_CHUNK] = top
        conf_flip[i:i + ROW_CHUNK] = (ta != tb) & (top > 0.5)
        d = lb[i:i + ROW_CHUNK].astype(np.float64) - lc[i:i + ROW_CHUNK].astype(np.float64)
        rms[i:i + ROW_CHUNK] = np.sqrt((d * d).mean(axis=-1))
        dc = d - d.mean(axis=-1, keepdims=True)  # a constant logit shift does not change the softmax
        rmsc[i:i + ROW_CHUNK] = np.sqrt((dc * dc).mean(axis=-1))
    return {"kld": kld, "agree": agree, "conf_flip": conf_flip, "logit_rms": rms, "logit_rms_centered": rmsc,
            "p1": p1}, np.arange(start, start + n)


def parse_shape(s):
    kind, dims = s.split(":")
    return kind, [int(d) for d in dims.split("x")]


def compare_tensors(bdir, cdir, mb, mc):
    out = {}
    for name, info in mb.get("tensors", {}).items():
        if (info["shape"] in ("noncontiguous", "unsupported") or name not in mc.get("tensors", {})
                or " " in name):  # "<name> (view)" etc. from older dumps duplicate <name>
            continue
        kind, dims = parse_shape(info["shape"])
        cnt = info["count"]
        if mc["tensors"][name]["count"] != cnt:
            out[name] = {"error": f"count {cnt} vs {mc['tensors'][name]['count']}"}
            continue
        dt = np.int32 if kind == "i32" else np.float32
        a = np.fromfile(bdir / f"t.{name}.bin", dtype=dt).reshape(cnt, -1)
        b = np.fromfile(cdir / f"t.{name}.bin", dtype=dt).reshape(cnt, -1)
        if kind == "i32":  # expert ids: compare as sets per position
            diff = np.array([len(set(x) ^ set(y)) // 2 for x, y in zip(a, b)])
            out[name] = {"kind": "ids", "flip": diff > 0, "n_diff": diff}
        else:
            a64, b64 = a.astype(np.float64), b.astype(np.float64)
            num = np.sqrt(((b64 - a64) ** 2).sum(axis=1))
            den = np.sqrt((a64 ** 2).sum(axis=1))
            out[name] = {"kind": "f32", "rel": num / np.maximum(den, 1e-30),
                         "max_abs": np.abs(b64 - a64).max(axis=1)}
    return out


def compare_kv(bdir, cdir):
    pa, pc = bdir / "kv.bin", cdir / "kv.bin"
    if not (pa.exists() and pc.exists()):
        return None
    a = np.fromfile(pa, dtype=np.uint8)
    b = np.fromfile(pc, dtype=np.uint8)
    if a.size != b.size:
        return {"error": f"size {a.size} vs {b.size}"}
    n = a.size // 2 * 2
    fa = a[:n].view(np.float16).astype(np.float64)
    fb = b[:n].view(np.float16).astype(np.float64)
    ok = np.isfinite(fa) & np.isfinite(fb) & (np.abs(fa) < 6e4)
    same_bits = a[:n].view(np.uint16) == b[:n].view(np.uint16)
    d = np.where(ok, fb - fa, 0.0)
    den = np.sqrt((np.where(ok, fa, 0.0) ** 2).sum())
    return {"bytes": int(a.size), "words_differ_pct": float(100 * (~same_bits).mean()),
            "rel_l2": float(np.sqrt((d ** 2).sum()) / max(den, 1e-30))}


def layer_of(name):
    """"ffn_moe-12" -> ("ffn_moe", 12); a repeated name within one step, "Qcur-2#1" -> ("Qcur#1", 2)."""
    m = re.match(r"(.*)-(\d+)(#\d+)?$", name)
    return (m.group(1) + (m.group(3) or ""), int(m.group(2))) if m else (name, -1)


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return None
    return {"mean": float(x.mean()), "median": float(np.median(x)), "p99": float(np.percentile(x, 99)),
            "max": float(x.max()), "n": int(x.size)}


def compare_runs(base, cand, per_position=False):
    base, cand = Path(base), Path(cand)
    prompts = sorted(p.name for p in base.iterdir() if (p / "meta.json").exists())
    per_prompt, all_pos, cont_pos, pos_out = {}, [], [], {}
    fam = {}
    for name in prompts:
        bdir, cdir = base / name, cand / name
        if not (cdir / "meta.json").exists():
            raise SystemExit(f"candidate is missing prompt {name}")
        mb, mc = read_meta(bdir), read_meta(cdir)
        tb, tc = np.fromfile(bdir / "tokens.i32", np.int32), np.fromfile(cdir / "tokens.i32", np.int32)
        if not np.array_equal(tb, tc):
            raise SystemExit(f"{name}: token sequences differ (was the candidate run with --tokens-from?)")
        lg, positions = compare_logits(bdir, cdir, mb, mc)
        if positions.size == 0:
            raise SystemExit(f"{name}: the two runs logged no common positions (check --logits-from / --prefill)")
        cont = positions >= mb["n_prompt"] - 1
        all_pos.append(lg)
        cont_pos.append({k: v[cont] for k, v in lg.items()})
        t = compare_tensors(bdir, cdir, mb, mc)
        flips = [r["flip"] for tn, r in t.items() if layer_of(tn)[0] == "ffn_moe_topk" and "flip" in r]
        if flips:  # tensor rows are per decoded position (0..n_tokens-1); logits rows start at logits_from
            nflip = np.sum(flips, axis=0)[positions]
            lg["route_flips"] = nflip
            lg["clean"] = np.cumsum(np.sum(flips, axis=0))[positions] == 0
        for tn, r in t.items():
            if "error" in r:
                fam.setdefault(("error", tn), []).append(r["error"])
                continue
            f, il = layer_of(tn)
            fam.setdefault((f, il), []).append(r)
        worst = int(np.argmax(lg["kld"]))
        if per_position:
            pp = {"pos": positions.tolist(), "kld": [float(f"{v:.4g}") for v in lg["kld"]],
                  "agree": lg["agree"].astype(int).tolist(), "p1": [round(float(v), 4) for v in lg["p1"]]}
            if "route_flips" in lg:
                pp["route_flips"] = lg["route_flips"].astype(int).tolist()
            pos_out[name] = pp
        per_prompt[name] = {"n_prompt": mb["n_prompt"], "n_positions": int(lg["kld"].size),
                            "kld_mean": float(lg["kld"].mean()), "kld_max": float(lg["kld"].max()),
                            "kld_max_pos": int(positions[worst]), "top1_pct": float(100 * lg["agree"].mean()),
                            "cand_first_argmax_diff": mc.get("first_argmax_diff", -1),
                            "kv": compare_kv(bdir, cdir)}

    def agg(parts):
        keys = set.intersection(*(set(p) for p in parts))
        cat = {k: np.concatenate([p[k] for p in parts]) for k in keys}
        if cat["kld"].size == 0:
            return None
        r = {"kld": stats(cat["kld"]), "top1_pct": float(100 * cat["agree"].mean()),
             "confident_flip_pct": float(100 * cat["conf_flip"].mean()),
             "logit_rms": stats(cat["logit_rms"]), "logit_rms_centered": stats(cat["logit_rms_centered"]),
             "n_positions": int(cat["kld"].size)}
        if "clean" in cat:
            c = cat["clean"].astype(bool)
            r["routing"] = {"clean_pct": float(100 * c.mean()),
                            "kld_clean": stats(cat["kld"][c]), "kld_after_flip": stats(cat["kld"][~c]),
                            "top1_clean_pct": float(100 * cat["agree"][c].mean()) if c.any() else None,
                            "top1_after_flip_pct": float(100 * cat["agree"][~c].mean()) if (~c).any() else None}
        return r

    tensors = {}
    for (f, il), rs in sorted(fam.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if f == "error":
            tensors.setdefault("errors", {})[il] = rs
            continue
        e = tensors.setdefault(f, {"per_layer": {}})
        if rs[0]["kind"] == "ids":
            flip = np.concatenate([r["flip"] for r in rs])
            nd = np.concatenate([r["n_diff"] for r in rs])
            e["per_layer"][str(il)] = {"flip_pct": float(100 * flip.mean()), "mean_experts_changed": float(nd.mean())}
        else:
            rel = np.concatenate([r["rel"] for r in rs])
            e["per_layer"][str(il)] = {"rel_mean": float(rel.mean()), "rel_p99": float(np.percentile(rel, 99)),
                                  "rel_max": float(rel.max())}
    for f, e in tensors.items():
        if f == "errors":
            continue
        vals = list(e["per_layer"].values())
        if "flip_pct" in vals[0]:
            e["flip_pct"] = float(np.mean([v["flip_pct"] for v in vals]))
            e["mean_experts_changed"] = float(np.mean([v["mean_experts_changed"] for v in vals]))
        else:
            e["rel_mean"] = float(np.mean([v["rel_mean"] for v in vals]))
            e["rel_max"] = float(max(v["rel_max"] for v in vals))
    kvs = [p["kv"]["rel_l2"] for p in per_prompt.values() if p["kv"] and "rel_l2" in p["kv"]]
    # where greedy decoding would first leave the reference's tokens (teacher-forced candidates; reported, not gated)
    div = [p["cand_first_argmax_diff"] - p["n_prompt"] for p in per_prompt.values() if p["cand_first_argmax_diff"] >= 0]
    greedy = {"identical_prompts": len(per_prompt) - len(div), "prompts": len(per_prompt),
              "median_divergence": float(np.median(div)) if div else None, "earliest_divergence": min(div) if div else None}
    res = {"base": str(base), "cand": str(cand), "all": agg(all_pos), "continuation": agg(cont_pos),
           "kv_rel_l2_max": max(kvs) if kvs else None, "greedy": greedy, "tensors": tensors, "prompts": per_prompt}
    if per_position:
        res["per_position"] = pos_out
    return res


def check_policy(res, policy):
    """policy: {"metric.path": {"max": x} | {"min": x}}; paths index into the result, e.g. "continuation.kld.mean"."""
    fails, checked = [], {}
    for path, lim in policy.items():
        if path.startswith("_"):
            continue
        v = res
        for k in path.split("."):
            v = v.get(k) if isinstance(v, dict) else None
        checked[path] = v
        if v is None:
            fails.append(f"{path}: missing")
        elif "max" in lim and v > lim["max"]:
            fails.append(f"{path} = {v:.4g} > {lim['max']:.4g}")
        elif "min" in lim and v < lim["min"]:
            fails.append(f"{path} = {v:.4g} < {lim['min']:.4g}")
    return {"pass": not fails, "fails": fails, "checked": checked}


def fmt(res):
    lines = [f"base {res['base']}\ncand {res['cand']}"]
    for k in ("all", "continuation"):
        a = res[k]
        if not a:
            continue
        lines.append(f"{k:>12}: n={a['n_positions']}  KLD mean {a['kld']['mean']:.3g} median {a['kld']['median']:.3g} "
                     f"p99 {a['kld']['p99']:.3g} max {a['kld']['max']:.3g}  top-1 {a['top1_pct']:.2f}%  "
                     f"confident flips {a['confident_flip_pct']:.2f}%  logit RMS {a['logit_rms']['mean']:.3g}")
        if "routing" in a:
            r = a["routing"]
            fmt_k = lambda s: f"{s['mean']:.3g} (n={s['n']})" if s else "-"
            lines.append(f"{'':>14}routing identical so far at {r['clean_pct']:.1f}% of positions: KLD mean there "
                         f"{fmt_k(r['kld_clean'])}, after a routing difference {fmt_k(r['kld_after_flip'])}")
    for f, e in res["tensors"].items():
        if f == "errors":
            lines.append(f"  tensor errors: {e}")
        elif "flip_pct" in e:
            lines.append(f"  {f:<22} routing flips {e['flip_pct']:.2f}% of layer-positions "
                         f"({e['mean_experts_changed']:.3f} experts changed on average)")
        else:
            worst = max(e["per_layer"].items(), key=lambda kv: kv[1]["rel_mean"])
            lines.append(f"  {f:<22} rel L2 mean {e['rel_mean']:.3g} max {e['rel_max']:.3g} "
                         f"(worst layer {worst[0]}: {worst[1]['rel_mean']:.3g})")
    g = res.get("greedy")
    if g:
        tail = (f", the rest diverge after median {g['median_divergence']:.0f} (earliest {g['earliest_divergence']}) "
                f"generated tokens" if g["median_divergence"] is not None else "")
        lines.append(f"  greedy text identical in {g['identical_prompts']} of {g['prompts']} prompts{tail}")
    for p, r in res["prompts"].items():
        kv = r["kv"]
        kvs = f" kv rel {kv['rel_l2']:.3g} ({kv['words_differ_pct']:.1f}% words differ)" if kv and "rel_l2" in kv else ""
        lines.append(f"  {p:<24} KLD mean {r['kld_mean']:.3g} max {r['kld_max']:.3g} @{r['kld_max_pos']}  "
                     f"top-1 {r['top1_pct']:.1f}%  greedy diverges at {r['cand_first_argmax_diff']}{kvs}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("cand")
    ap.add_argument("--json"); ap.add_argument("--policy"); ap.add_argument("--policy-set", default="default")
    ap.add_argument("--quiet", action="store_true"); ap.add_argument("--per-position", action="store_true")
    a = ap.parse_args()
    res = compare_runs(a.base, a.cand, a.per_position)
    if a.policy:
        pol = json.loads(Path(a.policy).read_text())
        res["policy"] = {"set": a.policy_set, **check_policy(res, pol[a.policy_set])}
    if not a.quiet:
        print(fmt(res))
        if "policy" in res:
            print("POLICY", "PASS" if res["policy"]["pass"] else "FAIL: " + "; ".join(res["policy"]["fails"]))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    sys.exit(0 if res.get("policy", {}).get("pass", True) else 3)


if __name__ == "__main__":
    main()
