for i in *.csv; do
    sed -i '/null/d' $i || break
done