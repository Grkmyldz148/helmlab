import { VERSION } from './version';

/** One-line highlights per release — feeds the auto-generated Twitter card.
 *  Add an entry when bumping the version; unknown versions fall back to
 *  evergreen stats so the card never ships empty. */
export const releaseHighlights: Record<string, string[]> = {
  '0.16.0': [
    'Confidence v2 — ordinal-probit observer model',
    'New pNoticeable: chance an observer sees the difference',
    'Python ↔ JS parity held at ~1e-15',
  ],
  '0.15.0': [
    'The parity release: full API, zero diffs at 1e-12',
    'genToLch / genFromLch for hue rotations & harmonies',
    'PCHIP neutral correction + trusted publishing',
  ],
};

export const currentHighlights: string[] =
  releaseHighlights[VERSION] ?? [
    'Two purpose-built Lab spaces: GenSpace + MetricSpace',
    'STRESS 22.48 on COMBVD — 23% better than CIEDE2000',
    'Python & JavaScript, identical output',
  ];
