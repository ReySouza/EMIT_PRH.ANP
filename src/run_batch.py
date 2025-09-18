# run_batch.py (atualizado)
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
from typing import Dict, List, Optional
import yaml

def tprint(*a): print("[RUN]", *a)

def load_cfg(p: Path) -> Dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def find_hdr_scenes(raw_dir: Path) -> List[Path]:
    return sorted(list(raw_dir.rglob("*.hdr")))

def stem_from_hdr(hdr: Path) -> str:
    return hdr.stem

def run_py(script: Path, args: List[str], cwd: Optional[Path]=None) -> int:
    cmd = [sys.executable, str(script)] + args
    tprint("EXEC:", " ".join([str(c) for c in cmd]))
    return subprocess.run(cmd, cwd=str(cwd or script.parent)).returncode

def main():
    ap = argparse.ArgumentParser(
        description="Pipeline em lote: MAG1C -> preprocess -> MAG1C patches -> mask_generation -> (opcional) train/eval/predict"
    )
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--reports-dir", default="reports")

    # novas flags para a etapa MAG1C
    ap.add_argument("--skip-mag1c", action="store_true", help="Pula a execução do mag1c.py por cena")
    ap.add_argument("--mag1c-g", type=int, default=None, help="Valor para o parâmetro -g do mag1c.py (default=4)")
    ap.add_argument("--mag1c-overwrite", action="store_true",
                help="Passa --overwrite ao mag1c.py para sobrescrever saídas existentes")

    # flags existentes
    ap.add_argument("--skip-preprocess", action="store_true")
    ap.add_argument("--skip-mag1c-patches", action="store_true", help="Pula a geração de patches MAG1C")
    ap.add_argument("--skip-masks", action="store_true", help="Pula mask_generation (rótulos)")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-predict", action="store_true")
    ap.add_argument("--scene", type=str, default=None, help="Opcional: processa apenas essa cena (stem do .hdr)")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_cfg(cfg_path)
    root = Path(".").resolve()
    src = root / "src" if (root / "src").exists() else root

    paths = cfg["paths"]
    raw_dir = Path(paths["raw_data_dir"]).resolve()
    outputs = Path(paths["output_dir"]).resolve()
    reports = ensure_dir(Path(args.reports_dir).resolve())

    # Descobrir cenas (.hdr)
    hdrs = find_hdr_scenes(raw_dir)
    if args.scene:
        hdrs = [h for h in hdrs if stem_from_hdr(h).startswith(args.scene)]
    if not hdrs:
        tprint(f"Nenhum .hdr encontrado em {raw_dir} para scene='{args.scene or '*'}'"); return

    scenes = [stem_from_hdr(h) for h in hdrs]
    tprint(f"{len(scenes)} cena(s):", ", ".join(scenes[:6]) + ("..." if len(scenes) > 6 else ""))

    # ============ [0] MAG1C (novo passo) ============
    # Executa mag1c.py para cada cena .hdr encontrada.
    # Entrada: caminho base SEM extensão (.hdr/.img).
    # Saída:   arquivos <stem>.hdr/.img gravados DIRETAMENTE em data/mag1c
    if not args.skip_mag1c:
        mag1c_script = root / "mag1c.py"  # no mesmo diretório do run_batch.py (src/)
        if not mag1c_script.exists():
            tprint("[WARN] mag1c.py não encontrado; pulando etapa MAG1C.")
        else:
            from shutil import rmtree
            mag1c_out_base = ensure_dir(Path(paths["mag1c_dir"]).resolve())
            # obter -g: CLI > YAML > default(4)
            g_val = int(args.mag1c_g) if args.mag1c_g is not None else int(cfg.get("mag1c", {}).get("g", 4))
            tprint(f"[MAG1C] Parâmetro -g = {g_val}")

            # evitar duplicatas por stem
            unique_by_stem: Dict[str, Path] = {}
            for hdr in hdrs:
                st = stem_from_hdr(hdr)
                if st not in unique_by_stem:
                    unique_by_stem[st] = hdr

            for st, hdr in unique_by_stem.items():
                # base de entrada SEM extensão
                hdr_str = str(hdr)
                in_base = hdr_str[:-4] if hdr_str.lower().endswith(".hdr") else hdr_str
                if not Path(in_base + ".hdr").exists():
                    tprint(f"[WARN] .hdr não encontrado para base: {in_base}; pulando {st}")
                    continue

                # base de saída SEM extensão PLANA: .../data/mag1c/<stem>
                out_base = mag1c_out_base / st  # NÃO cria subpasta

                # caso exista uma PASTA com o mesmo nome do stem (resto de execuções anteriores):
                if out_base.exists() and out_base.is_dir():
                    if args.mag1c_overwrite:
                        tprint(f"[MAG1C] Removendo diretório existente para achatar saída: {out_base}")
                        rmtree(out_base)
                    else:
                        tprint(f"[WARN] Existe um diretório com nome {st} em {mag1c_out_base}."
                               f" Remova-o ou use --mag1c-overwrite para eu limpar e gravar {st}.hdr/.img direto.")
                        continue

                # montar args
                mag_args = [str(in_base), "--out", str(out_base), "-g", str(g_val)]
                if args.mag1c_overwrite:
                    mag_args.append("--overwrite")

                tprint("EXEC:", sys.executable, str(mag1c_script), *mag_args)
                rc = run_py(mag1c_script, mag_args, cwd=mag1c_script.parent)
                if rc != 0:
                    tprint(f"[WARN] mag1c.py falhou para cena: {st}")



    # ============ [1] preprocess (gera patches + manifest/stats) ============
    if not args.skip_preprocess:
        for hdr in hdrs:
            rc = run_py(src/"preprocess.py", ["--config", str(cfg_path), "--hdr", str(hdr)], cwd=src)
            if rc != 0: tprint(f"[WARN] preprocess falhou: {hdr}")

    # ============ [2] MAG1C patches (gera patches 1-canal alinhados ao manifest) ============
    if not args.skip_mag1c_patches:
        mm = src / "mag1c_patches.py"
        if not mm.exists():
            tprint("[WARN] mag1c_patches.py não encontrado no repo; usando versão local (se existir).")
            mm = Path("mag1c_patches.py")
        for sc in scenes:
            rc = run_py(mm, ["--config", str(cfg_path), "--scene", sc], cwd=src)
            if rc != 0: tprint(f"[WARN] mag1c_patches falhou: {sc}")

    # ============ [3] mask_generation (gera máscara binária para treino) ============
    if not args.skip_masks:
        for sc in scenes:
            rc = run_py(src/"mask_generation.py", ["--config", str(cfg_path), "--scene", sc], cwd=src)
            if rc != 0: tprint(f"[WARN] mask_generation falhou: {sc}")

    # ============ [4] treino (opcional) — 1 vez geral ============
    if not args.skip_train:
        rc = run_py(src/"train.py", [str(cfg_path)], cwd=src)
        if rc != 0: tprint("[WARN] train falhou")

    # ============ [5] avaliação (opcional) ============
    if not args.skip_eval:
        rc = run_py(src/"evaluate.py", [str(cfg_path)], cwd=src)
        if rc != 0: tprint("[WARN] evaluate falhou")

    # ============ [6] predição por cena (opcional) ============
    if not args.skip_predict:
        for sc in scenes:
            rc = run_py(src/"predict.py", ["--config", str(cfg_path), "--scene", sc], cwd=src)
            if rc != 0: tprint(f"[WARN] predict falhou: {sc}")

    tprint("Fim.")

if __name__ == "__main__":
    main()
