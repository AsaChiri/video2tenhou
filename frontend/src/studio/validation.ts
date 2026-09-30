export function gameIds(text: string) {
  const parts = text
    .trim()
    .split(/[\s,]+/)
    .filter(Boolean);
  if (!parts.length)
    throw new Error("Enter the scoremj game IDs in recording order.");
  return parts.map((part: string) => {
    let value = part;
    if (part.startsWith("https://")) {
      const u = new URL(part);
      if (!/(^|\.)scoremj\.com$/.test(u.hostname))
        throw new Error("Use scoremj game links or numeric game IDs.");
      value =
        u.searchParams.get("id") ||
        u.pathname.match(/\d+\/?$/)?.[0]?.replace("/", "") ||
        "";
    }
    if (
      !/^\d+$/.test(value) ||
      Number(value) <= 0 ||
      !Number.isSafeInteger(Number(value))
    )
      throw new Error("Use positive game IDs, separated by commas or spaces.");
    return Number(value);
  });
}

export function timeRange(start: string, end: string) {
  const parse = (value: string) => {
    if (!value) return 0;
    if (!/^\d+(?::[0-5]\d){0,2}(?:\.\d+)?$/.test(value))
      throw new Error("Use seconds, MM:SS or HH:MM:SS.");
    return value
      .split(":")
      .reduce((total, part) => total * 60 + Number(part), 0);
  };
  if (end && parse(end) <= parse(start))
    throw new Error("End time must be after start time.");
  parse(start);
  return { start, end };
}
