# logdir.sh — sourced by submit_suites.sh and submit_docs.sh: new_logdir <base> <name> creates and prints (absolute) a
# fresh log directory <base>/<YYYYmmdd-HHMMSS>-<name>-<nonce>. The nonce (mktemp's) keeps two launches of the same commit
# in the same second apart; before it they shared one directory and overwrote each other's <Package>.log.
new_logdir() {
  local base=$1 name=$2 d
  mkdir -p "$base" || return 1
  d=$(mktemp -d "$base/$(date +%Y%m%d-%H%M%S)-$name-XXXXXX") && chmod 755 "$d" || return 1   # mktemp -d is 0700
  (cd "$d" && pwd)
}
