"""Reply-length spread (non-quiet turns) and two sample sessions of a
flow.dataset folder or an eval file: python scripts/lengths.py PATH.

Three views of the same replies:
  - the five buckets of the r4p pilot (comparable with its numbers);
  - the length classes of the beats3 revision 2 plan (flow.style.BANDS:
    short / medium / long / over) as shares of the non-quiet replies, to set
    against --length-mix (35 / 45 / 20);
  - when the records carry a length_target (a revision 2 dataset), the hit
    rate: the share of replies whose class is the target the teacher was
    asked for, in total and per target, and how the replies of each target
    came out."""
import sys, re, random, collections
sys.path.insert(0, '.')
from flow import checks, style as sty


def main(path):
    for lab, recs in checks.load(path):
        def size(s):
            cjk = len(re.findall(r'[぀-ヿ一-鿿]', s))
            return ('cjk', cjk) if cjk > len(s) * 0.3 else ('en', len(re.findall(r"[A-Za-z']+", s)))
        b = collections.Counter()
        cls = collections.Counter()
        by_t = collections.defaultdict(collections.Counter)
        for r in recs:
            if r['quiet']: continue
            k, n = size(r['spoken'])
            edges = (10, 25, 45, 70) if k == 'en' else (25, 60, 110, 170)
            b[(k, sum(n > e for e in edges))] += 1
            c = sty.length_class(r['spoken'])
            cls[c] += 1
            if r.get('target'):
                by_t[r['target']][c] += 1
        print(lab)
        for k, names in (('en', 'words ≤10 / ≤25 / ≤45 / ≤70 / >70'), ('cjk', 'chars ≤25 / ≤60 / ≤110 / ≤170 / >170')):
            tot = sum(v for (kk, _), v in b.items() if kk == k) or 1
            print(f"  {k} {names}: " + ' / '.join(f"{b[(k, i)] / tot:.0%}" for i in range(5)))
        tot = sum(cls.values()) or 1
        print("  length class short ≤20w/35c, medium ≤40/80, long ≤60/130, over: "
              + ' / '.join(f"{cls[c] / tot:.0%}" for c in sty.LENGTH_CLASSES)
              + f"   (n={tot}; target mix 35 / 45 / 20)")
        if by_t:
            nt = sum(sum(c.values()) for c in by_t.values())
            hit = sum(c[t] for t, c in by_t.items())
            print(f"  length target hit rate: {hit / nt:.0%} of {nt}")
            for t in sty.LENGTH_TARGETS:
                c = by_t.get(t)
                if c:
                    n = sum(c.values())
                    print(f"    asked {t:6s} (n={n}): " + ', '.join(
                        f"{k} {c[k] / n:.0%}" for k in sty.LENGTH_CLASSES))
            asked = collections.Counter({t: sum(c.values()) for t, c in by_t.items()})
            print("    asked mix: " + ' / '.join(f"{asked[t] / nt:.0%}" for t in sty.LENGTH_TARGETS))
        random.seed(1)
        for sid in random.sample(sorted({r['session'] for r in recs}), 2):
            print('\n== session', sid)
            for r in sorted((r for r in recs if r['session'] == sid), key=lambda r: r['turn'])[:6]:
                print('  viewer:', r['user'][:150].replace('\n', ' | '))
                print('  her   :', r['spoken'][:400], f"[{r['target']}]" if r.get('target') else '')


if __name__ == '__main__':
    main(sys.argv[1])
