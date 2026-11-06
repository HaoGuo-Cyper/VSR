"""Link existing local assets without copying or modifying their contents."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/home/hugo/Projects/VSR'))
    parser.add_argument('--pretrained-root', type=Path,
                        default=Path('/home/hugo/Projects/ReprVSR-pretrained'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = args.source.resolve(strict=True)
    pretrained = args.pretrained_root.resolve(strict=True)
    if root == source:
        raise SystemExit('Run this from the new checkout, not the original project.')
    links = {
        '.deps': source / '.deps',
        'weights': source / 'weights',
        'data': source / 'data',
        'vendor/dinov2': source / 'vendor/dinov2',
        'local-assets/original-runs': source / 'runs',
        'local-assets/HQ-VSR': pretrained / 'data/HQ-VSR',
        'local-assets/Wan2.1-T2V-1.3B-Diffusers': pretrained / 'models/Wan2.1-T2V-1.3B-Diffusers',
    }
    for name, target in links.items():
        target = target.resolve(strict=True)
        link = root / name
        if link.is_symlink() and link.resolve() == target:
            print(f'Already linked: {name}')
            continue
        if link.exists() or link.is_symlink():
            raise SystemExit(f'Refusing to replace existing path: {link}')
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=target.is_dir())
        print(f'Linked {name} -> {target}')
    # Deliberately keep new outputs separate from the original run directory.
    (root / 'runs').mkdir(exist_ok=True)


if __name__ == '__main__':
    main()
