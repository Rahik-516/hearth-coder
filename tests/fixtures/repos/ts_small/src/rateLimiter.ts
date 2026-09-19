/** Throttles outbound calls, mirroring the Python fixture's TokenBucket. */
export class RateLimiter {
  private tokens: number;
  private last: number;

  constructor(
    private readonly capacity: number,
    private readonly ratePerSecond: number,
  ) {
    this.tokens = capacity;
    this.last = Date.now();
  }

  private refill(): void {
    const now = Date.now();
    const elapsed = (now - this.last) / 1000;
    this.tokens = Math.min(this.capacity, this.tokens + elapsed * this.ratePerSecond);
    this.last = now;
  }

  tryAcquire(count = 1): boolean {
    this.refill();
    if (this.tokens >= count) {
      this.tokens -= count;
      return true;
    }
    return false;
  }

  acquire(count = 1): void {
    if (!this.tryAcquire(count)) {
      throw new Error("rate limit exceeded");
    }
  }
}
