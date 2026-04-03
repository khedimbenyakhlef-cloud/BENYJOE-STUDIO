#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# build_zip.sh — Packaging du méga-zip BENYJOE_STUDIO_v10
# Usage : bash build_zip.sh
# ═══════════════════════════════════════════════════════════════
set -e

PROJET="BENYJOE_STUDIO_v10"
ARCHIVE="BENYJOE_STUDIO_v10_MEGA.zip"

echo "╔══════════════════════════════════════════════╗"
echo "║   BENY-JOE STUDIO — Build méga-zip v10.1    ║"
echo "╚══════════════════════════════════════════════╝"

# Nettoyage de l'ancienne archive si elle existe
if [ -f "$ARCHIVE" ]; then
    echo "[INFO] Suppression ancienne archive $ARCHIVE"
    rm -f "$ARCHIVE"
fi

echo "[INFO] Compression de $PROJET/ → $ARCHIVE …"
zip -r "$ARCHIVE" "$PROJET/" \
    --exclude "*.pyc" \
    --exclude "*/__pycache__/*" \
    --exclude "*/.git/*" \
    --exclude "*/node_modules/*" \
    --exclude "*/outputs/*.mp4" \
    --exclude "*/outputs/*.png" \
    --exclude "*/logs/*.log"

echo ""
echo "[OK] Archive créée : $ARCHIVE"
echo "     Taille : $(du -sh "$ARCHIVE" | cut -f1)"
echo ""
echo "[TEST] Vérification de l'intégrité …"
unzip -t "$ARCHIVE" > /dev/null && echo "[OK] Archive valide ✓"

echo ""
echo "Contenu :"
unzip -l "$ARCHIVE" | tail -n +4 | head -40
