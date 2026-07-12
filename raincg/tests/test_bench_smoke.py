from raincg.bench.run_benchmark import render_table, BenchRow


def test_render_table_contains_rows():
    rows = [BenchRow("VSA (pure_vsa)", 0, "1.0s", "5.0s", 0.9998, 9719, 9721),
            BenchRow("Transformer", 8_400_000, "42min", "120s", 0.86, 8360, 9721)]
    txt = render_table(rows)
    assert "VSA (pure_vsa)" in txt
    assert "Transformer" in txt
    assert "99.98%" in txt
    assert "9719/9721" in txt
