# Complete raw Meep evidence

All 45 original scientific files are preserved losslessly in `raw-meep-v4.tar.gz`, including raw NPZ monitors, JSON results, stdout/stderr, the journal, plan and runtime identity manifest. `RAW_MANIFEST.json` pins the archive and every individual file with SHA256. Readable plan, input, reference and summary copies are alongside it. Compression keeps numeric JSON out of a source-code diff without discarding data.

Extract the archive into a fresh directory to reproject the completed historical run. Historical journal commands retain their original execution paths. Use `optional/meep_verified/README.md` for a new validation run; another host must freeze its own runtime identity. All frozen scientific bytes and hashes remain intact.

The 535-check independent coding-agent audit is in `docs/boundary/review/meep-v4`. This evidence does not establish external human validation, adoption or physical-device performance.
