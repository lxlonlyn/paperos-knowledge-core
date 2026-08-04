# Purpose

Extract claims and evidence links without replacing source text.

# Template

Given versioned canonical chunks, return structured claims with their exact
supporting canonical chunk IDs, claim type, and confidence. Do not alter source
text or cite IDs outside the supplied snapshot. Every extracted claim must retain
its derivation chain.
