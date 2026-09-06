"""Statistical tests and effect sizes.

Pre-registered tests are always run and always reported. Where a pre-registered test
is the wrong one for the data, the correct test is reported *alongside* it, never
instead of it [M4].
"""

from cosq.stats.effects import (
    bootstrap_ci,
    cliffs_delta,
    mcnemar_odds_ratio,
    risk_difference,
)
from cosq.stats.multiple import holm_bonferroni
from cosq.stats.power import (
    DiscordanceReport,
    discordance_report,
    mcnemar_power,
    required_discordant_pairs,
)
from cosq.stats.tests import (
    TestResult,
    cochran_q,
    mcnemar_exact,
    shapiro_wilk,
    wilcoxon_signed_rank,
)

__all__ = [
    "DiscordanceReport",
    "TestResult",
    "bootstrap_ci",
    "cliffs_delta",
    "cochran_q",
    "discordance_report",
    "holm_bonferroni",
    "mcnemar_exact",
    "mcnemar_odds_ratio",
    "mcnemar_power",
    "required_discordant_pairs",
    "risk_difference",
    "shapiro_wilk",
    "wilcoxon_signed_rank",
]
