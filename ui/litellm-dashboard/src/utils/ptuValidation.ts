interface ValidatorRule {
  validator: (rule: unknown, value: unknown) => Promise<void>;
}

interface FormInstance {
  getFieldValue: (name: string) => unknown;
}

export const PTU_COUNT_FIELD = "ptu_count";
export const PTU_RATE_FIELD = "cost_per_ptu_per_hour";

const isFilled = (value: unknown): boolean => value !== undefined && value !== null && value !== "";

const isPositiveWholeNumber = (value: unknown): boolean => {
  if (!isFilled(value)) {
    return true;
  }
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0;
};

/** Mirrors the backend contract, which rejects a ptu_count that is not a positive integer. */
export const ptuCountRules: ValidatorRule[] = [
  {
    validator: (_, value) =>
      isPositiveWholeNumber(value)
        ? Promise.resolve()
        : Promise.reject(new Error("PTU Count must be a positive whole number")),
  },
];

/**
 * The backend rejects a half-set pair with "ptu_count and cost_per_ptu_per_hour must be set
 * together", so filling or clearing one field without the other is caught in the form. Pair
 * this with `dependencies` on the sibling field so its error clears when the pair is resolved.
 */
export const ptuPairRule =
  (siblingField: string) =>
  ({ getFieldValue }: FormInstance): ValidatorRule => ({
    validator: (_, value) =>
      isFilled(value) === isFilled(getFieldValue(siblingField))
        ? Promise.resolve()
        : Promise.reject(new Error("PTU Count and Cost per PTU / Hour must be set together")),
  });
