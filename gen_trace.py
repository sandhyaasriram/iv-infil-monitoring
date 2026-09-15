import numpy as np
import re

labels = ['"normal"']*25 + ['"early"']*25 + ['"moderate"']*25 + ['"severe"']*25
T = [32.95]*25 + [32.07]*25 + [31.23]*25 + [30.73]*25
P = [1162.2]*25 + [1178.2]*25 + [1215.0]*25 + [1268.9]*25

np.random.seed(42)
rows = []
for i in range(100):
    t_noise = T[i] + np.random.normal(0, 0.05)
    p_noise = P[i] + np.random.normal(0, 1.0)
    rows.append(f'  {{{t_noise:.2f}, {p_noise:.1f}, {labels[i]}}}')

rows_str = ',\n'.join(rows)
replacement = f"const FallbackRow FALLBACK_TRACE[] = {{\n{rows_str}\n}};"

with open('firmware/sketch/sketch.ino', 'r') as f:
    code = f.read()

code = re.sub(r'const FallbackRow FALLBACK_TRACE\[\] = \{[^}]*\};', replacement, code, flags=re.MULTILINE)

with open('firmware/sketch/sketch.ino', 'w') as f:
    f.write(code)

print("Replaced FALLBACK_TRACE with flat 100-element array in sketch.ino")
