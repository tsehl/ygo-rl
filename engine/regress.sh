#!/usr/bin/env bash
# Non-régression complète contre l'oracle (graines de ETAPE2.md), puis obs_check, vmap_check et sb_check (banque de test, puis
# banque procédurale 2).
# Usage : engine/regress.sh [dossier_logs] ; JOBS=n processus JAX en parallèle (défaut : 4).
# Résumé à la fin : une ligne « k/n parties identiques » par mode.
set -uo pipefail
status=0
cd "$(dirname "$0")"
out=${1:-/tmp/ygo-regress}
mkdir -p "$out"
jobs=${JOBS:-4}  # ~1,7 Go par compare.py (pic de compilation)
run=${YGO_PYTHON:-"uv run --quiet --with jax python"}

# Une seule file pour tout : les vérifications longues (sb, sbp : ~16 min, pic ~5,8 Go) d'abord, puis les modes de compare.py.
printf '%s\n' "sbp" "sb" "vmap" "obs" "trig 100 1000" "quick 80 1600" "cont 100 3000" "dmg 100 4000" "ds 100 5000" "flag 100 6000" "neg 100 7000" "fus 100 8000" "rit 100 9000" "syn 100 10000" "xyz 100 11000" "lnk 100 12000" "xlk 100 13000" "kn 100 14000" "pen 100 15000" "chain 60 1200" "mixed 60 1300" "normal 40 1" \
  | xargs -P "$jobs" -L 1 sh -c '
      case "$0" in
        sbp) SB_PROC=2 '"$run"' sb_check.py ;;   # banque procédurale : fusions, filtres d ATK
        sb) '"$run"' sb_check.py ;;
        vmap) '"$run"' vmap_check.py ;;
        obs) '"$run"' obs_check.py ;;
        *) '"$run"' compare.py "$1" "$2" "$0" ;;
      esac > '"$out"'/"$0".log 2>&1' || status=1

for m in trig quick cont dmg ds flag neg fus rit syn xyz lnk xlk kn pen chain mixed normal; do
  printf '%-7s %s\n' "$m" "$(grep -h 'parties identiques' "$out/$m.log" || echo 'ÉCHEC (voir le log)')"
done
printf '%-7s %s\n' obs "$(tail -n 1 "$out/obs.log")"
printf '%-7s %s\n' vmap "$(tail -n 1 "$out/vmap.log")"
printf '%-7s %s\n' sb "$(tail -n 1 "$out/sb.log")"
printf '%-7s %s\n' sbp "$(tail -n 1 "$out/sbp.log")"

exit "$status"
