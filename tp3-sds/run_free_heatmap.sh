#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

OUT="$(cd .. && pwd)/tp3-output-2_5"
JAVA="java -Dframes.enabled=false -Doutput.dir=$OUT -cp out simulation.Main"
SEED=42

run() { # t1 t2 k mod simTime
  echo "→ mod=$4 t1=$1 t2=$2 k=$3 simTime=$5"
  $JAVA "$1" "$2" "$3" "$4" FREE "$5" "$SEED" >/dev/null
}

for MOD in A B; do
  for T1 in 0.5 1.0 1.5 2.0 2.5 3.0; do
    for T2 in 1.0 2.0 3.0 4.0 5.0 6.0; do
      run "$T1" "$T2" 5 "$MOD" 1000
    done
  done
done

echo "Done."
