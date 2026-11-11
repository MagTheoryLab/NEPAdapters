"""Serialize frozen independent JSON oracles for the dependency-free C++ test."""
import hashlib
import json
from pathlib import Path


def convert(root):
    for path in sorted(root.glob('*/*.json')):
        if path.parent.name not in ('soc', 'nosoc'):
            continue
        record = json.loads(path.read_text())
        model = path.with_suffix('.nep')
        assert hashlib.sha256(model.read_bytes()).hexdigest() == record['model_sha256']
        active = record.get('active', [True, False, False])
        lines = ['spin_density_oracle_v1', 'model_sha256 ' + record['model_sha256'],
                 'descriptor_dim ' + str(record['dimension']),
                 'active ' + ' '.join(str(int(v)) for v in active)]
        cases = record['cases']
        cases = cases.items() if isinstance(cases, dict) else ((c['name'], c) for c in cases)
        for name, case in cases:
            lines.extend([f'case {name} {len(case["types"])}', 'types ' + ' '.join(case['types'])])
            for key, source in [('cell', 'cell'), ('positions', 'positions'), ('spins', 'spins'),
                                ('potential', 'potential'), ('forces', 'force'), ('virial', 'virial'),
                                ('mforces', 'mforce'), ('tau', None), ('descriptor', None)]:
                if key == 'cell':
                    values = [case['cell'][3*j+i] for i in range(3) for j in range(3)]
                elif source:
                    values = case[source]
                    if values and isinstance(values[0], list): values = sum(values, [])
                elif key == 'descriptor' and 'descriptor' in case:
                    scalers = record['scaler'][30:]
                    values = []
                    for row in case['descriptor']:
                        values.extend([0.] * 30 + [v*s for v, s in zip(row, scalers)])
                else:
                    values = []
                lines.append(' '.join([key, str(len(values)), *(format(v, '.17g') for v in values)]))
            lines.append('end')
        path.with_suffix('.dat').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    convert(Path(__file__).parent / 'fixtures/spin_density')
