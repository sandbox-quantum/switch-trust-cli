.PHONY: release-audit license-check sbom vuln-scan lock-deps

release-audit: lock-deps license-check sbom vuln-scan
	@echo ""
	@echo "=== Release audit complete ==="
	@echo "Outputs in release/:"
	@ls -1 release/*.json release/*.csv release/*.txt 2>/dev/null
	@echo ""

# Packages whose license field contains the full license text rather than a
# short SPDX identifier. Manually verified:
#   replicate  — Apache License 2.0
#   tiktoken   — MIT License
#   zalgolib   — MIT License
#   fqdn       — MPL 2.0
#                (uses identifier "MPL-2.0" instead of "MPL 2.0" since
#                 https://github.com/ypcrts/fqdn/commit/
#                 9782faa6889f9797831d14454cc77830b7dea17f)
#   switch-trust-cli   — Apache License 2.0 (this project, license not yet in metadata)
IGNORE_PACKAGES := --ignore-packages replicate tiktoken zalgolib fqdn switch-trust-cli

license-check:
	@echo "=== License review ==="
	mkdir release/
	pip-licenses \
		--format=csv \
		--with-urls \
		--with-authors \
		--output-file=release/licenses.csv
	pip-licenses \
		--format=json \
		--with-urls \
		--with-authors \
		--output-file=release/licenses.json
	@echo "License summary:"
	@pip-licenses --summary --order=count
	@echo ""
	@echo "Checking for copyleft/restricted licenses..."
	@pip-licenses $(IGNORE_PACKAGES) --allow-only="Apache Software License;\
Apache Software License; BSD License;\
Apache Software License; MIT License;\
Apache License 2.0;\
Apache 2.0;\
Apache-2.0;\
Apache-2.0 AND CNRI-Python;\
Apache-2.0 AND MIT;\
Apache-2.0 OR BSD-2-Clause;\
Apache-2.0 OR BSD-3-Clause;\
Apache-2.0 OR MIT;\
Apache 2.0 License;\
Apache License;\
Apache;\
0BSD; \
BSD License;\
BSD-2-Clause;\
BSD-3-Clause;\
BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0;\
Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT;\
BSD 3-Clause OR Apache-2.0;\
3-Clause BSD License;\
MIT License;\
MIT;\
MIT OR Apache-2.0;\
MIT-CMU;\
MIT-0;\
ISC License (ISCL);\
ISC;\
Mozilla Public License 2.0 (MPL 2.0);\
MPL-2.0 AND (Apache-2.0 OR MIT);\
MPL-2.0 AND MIT;\
Python Software Foundation License;\
PSF;\
PSF-2.0;\
Public Domain;\
The Unlicense (Unlicense);\
Historical Permission Notice and Disclaimer (HPND);\
GNU Lesser General Public License v2 or later (LGPLv2+);\
GNU Lesser General Public License v3 or later (LGPLv3+);\
Zope Public License;\
ZPL-2.1;\
UNKNOWN" \
		2>&1 || (echo "WARNING: Some packages have licenses that need manual review — see above" && exit 1)
	@echo ""
	@echo "Packages with UNKNOWN license (need manual verification):"
	@pip-licenses --format=csv | grep UNKNOWN || echo "  (none)"
	@echo ""
	@echo "All known licenses are Apache-2.0 compatible."

sbom:
	@echo "=== SBOM generation (CycloneDX) ==="
	cyclonedx-py environment \
		-o release/sbom.json \
		--of json \
		--pyproject pyproject.toml
	@echo "SBOM written to release/sbom.json"

vuln-scan:
	@echo "=== Dependency vulnerability scan ==="
	pip-audit --format=json --output=release/vulnerabilities.json || true
	pip-audit --desc 2>&1 | tee release/vulnerabilities.txt
	@echo ""

lock-deps:
	@echo "=== Locking dependencies ==="
	pip-compile --no-strip-extras --extra full --output-file=requirements.lock pyproject.toml
	@echo "Locked dependencies written to requirements.lock"
