# Ethics and public-dataset access note

## Short answer

The computational analysis can be run without collecting new participants, contacting
hospitals, or recruiting readers. The primary Kermany data and the RSNA probe are public,
de-identified secondary datasets.

That does not mean the paper should claim that ethics review is irrelevant. The correct
publication-safe position is:

1. confirm the dataset licenses and terms of use;
2. ask the author’s institution or target journal whether a secondary public-data study
   needs review; and
3. obtain a written ethics-exemption/waiver decision if the institution requires one.

Do not invent an approval number or write “ethics approval was not required” without a
documented institutional basis.

## Nepal-specific interpretation

The Nepal Health Research Council 2022 guideline lists analysis of public-domain data
and anonymous/non-identified data among situations for which researchers can apply for
exemption from review. An exemption is a decision by the responsible ethics body; it is
not the same as silently skipping all documentation.

JNHRC and Nepalese Medical Journal author instructions require an ethics approval or
appropriate exemption statement for original research submissions. The manuscript
should therefore include the committee/office name, decision date or reference number,
the public data sources, and a statement that no identifiable records were accessed.

## Suggested manuscript wording after documentation is obtained

> This was a secondary analysis of publicly available, de-identified datasets. No new
> participants were recruited and no identifiable clinical records were accessed. The
> study was reviewed by [institution/committee], which determined that the work was
> [exempt from ethical review / approved], reference [number], dated [date].

Replace the bracketed fields only after receiving the actual decision.

## Data provenance to preserve

- Dataset landing-page URL and download date.
- Dataset version or release identifier.
- License/terms-of-use text or a saved copy of the relevant notice.
- SHA-256 checksums for downloaded archives where possible.
- The project commit, configuration, seed, and preprocessing manifest.
- A statement that RSNA labels were mapped from `Normal` and `Lung Opacity` only, while
  `No Lung Opacity / Not Normal` was excluded.

This documentation supports an exemption request and publication transparency; it does
not replace the decision of the author’s institution or the selected journal.
